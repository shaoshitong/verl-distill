# Qwen-Image-2.1：REFLOW → DMD 接入计划

状态：2026-09-21 已按计划编写代码与测试，按用户要求未运行训练、推理或测试。使用方法及实现边界见 [Qwen 使用文档](../qwen_image21.md)。本文件保留设计分析，以下“必须测试/需验证”均仍为待执行验收。

## 已确认范围

目标框架：`/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill`。
数据根目录：`/mnt/hdfs/__MERLIN_USER_DIR__/Qwen-Image-2.1-test/dataset_inference_20260921`。
模型根目录：`/mnt/hdfs/__MERLIN_USER_DIR__/Qwen-Image-2.1-test/model`。
参考图来自 dataset.zip 的解压目录，路径应配置化，不假设训练机器已有推理节点的 /tmp 文件。

- REFLOW：1000 次 Generator optimizer update。
- DMD：3000 次 Fake-score optimizer update，每 5 次 Fake-score 更新后做 1 次 Generator 更新，共 600 次 Generator 更新。
- 按用户最新说明，REFLOW 默认梯度累积 GA=1（允许后续显式配置）；DMD 的每次 Fake/Generator optimizer update 固定 GA=4。分别配置与记录两个阶段的 GA。
- 使用 FSDP1；不照搬 Seedream 的 FSDP64/SP4/EP8/OE64、网络变量或模型路径。
- 6-NFE 训练使用六个离散 timestep 上的加噪→去噪预测，不默认采用每次更新全程反传六步 rollout。完整六步串行去噪用于评测。
- 保留文生图和多参考图编辑，保留原 prompt、参考图顺序、尺寸、seed 和 latent 坐标定义。
- 不修改正在生产的推理代码、分片或输出文件，不停止现有出图任务。
- 保留框架当前已有的未提交修改，不重置或覆盖已有实验。

## 1. 先固定数据与模型接口

构建独立、不可变的训练 manifest，只收入存在有效 complete.json 的样本。快照时生产任务可以继续，但同一训练运行不自动吸收新样本；后续扩充必须生成新 manifest。

每条记录包括 kind/id、prompt、全部 reference_images、width/height、seed、initial_noise.pt、x0_latent.pt、image.png、模型 revision、推理参数和 payload SHA256。校验路径边界、唯一性、文件完整性、模型版本及尺寸，记录 manifest SHA256。提供全量 hash 校验与明确标记的快速索引模式，不把快速索引当作全量验证。

latent 为 diffusion-normalized `(1,64,1,H/16,W/16)`，transformer 输入为 `(1,HW/256,64)`。不能再次用 VAE mean/std 归一化训练目标；mean/std 只用于编码参考图或解码生成 latent。

新增 Qwen adapter，复用固定 Diffusers 提交的 prompt 编码、参考图 RGBA 处理、2048 等面积预处理、VAE 编码、image_pad_mask、img_shapes 与 timestep 约定。参考图 latent 排在目标 latent 前，仅对目标区域计算损失。

训练条件缓存与模型/处理器版本、prompt、参考图内容、预处理设置绑定，避免缓存失配。缓存可提前构建，避免每个训练 microbatch 同时常驻三份 transformer、VLM 和 VAE。训练有梯度的路径不跨更新复用 KV cache。

## 2. 六个 timestep 的定义

从 Qwen 模型实际 scheduler_config.json 构建六步网格，按目标 latent token 数计算动态 shift，沿用官方 set_timesteps 的变换顺序；按用户最新指定，仅将 Generator/REFLOW/六步评测使用的 shift_terminal 覆盖为0.4。不手写 Seedream shift=16 或 shift=4，不修改共享原始模型配置。

原始模型声明：use_dynamic_shifting=true、base_image_seq_len=256、max_image_seq_len=8192、base_shift=0.5、max_shift=0.9、shift_terminal=0.02、1000 个训练 timestep。本训练方案对Generator网格明确覆盖shift_terminal=0.4，score scheduler独立配置，不能被一并改为0.4。不同尺寸对应的网格可以不同。

必须测试：训练使用的六个 sigma 与同尺寸、相同shift_terminal=0.4覆盖设置下的六步推理网格一致；终点、time 输入缩放及 flow 符号一致。默认从这六个点采样执行一次加噪→预测，不宣称六个单独训练点本身等价于已验证的六步推理质量。

首版采用六点均匀采样，每点概率1/6，各rank独立抽样；不机械扩展Z-Image四点几何分布。配置sampling=uniform，日志记录命中数及各点loss，依据见第7节。分层或偏高噪声采样尚未实现。DMD 的 Generator 输入同样取六点；score 自身的噪声时间独立采样，不把 score 网格也限制为六点。

## 3. REFLOW：1000 次更新

令 z 为保存的 initial_noise，x0 为保存的最终 latent；采样六点之一 sigma，构造 xt=(1-sigma)*x0+sigma*z，目标 velocity=z-x0，只在目标图 latent 上计算 flow MSE。

参考当前 Z-Image ODE warmup 的有效参数：Generator LR=1e-4、betas=(0.9,0.999)、weight_decay=1e-2、clip=1、loss weight=1；默认 GA=1。先采用标准 AdamW，将 REFLOW 与 DMD optimizer 独立管理。

现有 Qwen 数据没有 Z-Image reward/source-rank 分数，因此默认样本权重为 1，不能伪造或继承缺失的 reward ranking。

保存完整 REFLOW 终点 checkpoint、训练配置、数据 hash 和六步评测结果，作为 DMD 起点。

## 4. DMD：3000 Fake / 600 Generator

首版初始化：Generator继承REFLOW终点；Fake-score从该终点建立独立参数副本；Real-score使用原始Qwen-Image-2.1 teacher并保持冻结。配置fake_initialization=reflow_generator明确记录本次选择。

每个 outer：5 次 Fake-score update + 1 次 Generator update，每次均使用 4 个 loader microbatch。Fake 阶段不更新 Generator；Generator 阶段不更新 Fake/Real 参数；标准输出层 DMD，无 GAN、feature alignment、schedule-free 或 EMA。

| 参数 | DMD 设置 |
|---|---|
| Generator optimizer | AdamW，LR=5e-7，betas=(0.9,0.999)，weight_decay=0.1 |
| Fake optimizer | AdamW，LR=5e-6，betas=(0.9,0.95)，weight_decay=0.1 |
| LR scheduler / warmup | 无；切阶段重建 optimizer，避免混用 REFLOW 动量 |
| Fake loss | epsilon regression，weight=1；按 flow 参数化明确变换关系 |
| Generator loss | 标准 DMD surrogate，weight=4，normalization epsilon=1e-6 |
| Gradient clip | Generator/Fake 均为 1 |
| Score time | 1000 点网格，[0.02,0.98]，具体边界与 shift 顺序写入测试 |
| Fake loss stabilization | 显式Qwen定义：min(((1-s)/max(s,1e-4))²,50)乘x0 MSE；未核实attempt25原函数，不宣称精确复现 |
| Guidance | 默认保持 Qwen 生产路径的 conditional prediction / true_cfg_scale=1，无额外 CFG=8 |
| Precision | BF16 前向参数计算，FP32 reduction/gradient/损失计算，无 GradScaler |

Generator 训练样本按六个离散时间点先加噪，再预测生成 x0；score 对生成结果独立加噪。六点采样训练与六步评测分别实现、分别验证。

每个 outer 消费 24 个 microbatch/每个数据并行 rank。有效样本数=24×micro_batch_size×数据并行 world_size；单次更新=4×micro_batch_size×数据并行 world_size。不将 SP/EP 数量重复计为样本数。

## 5. 框架接入、并行与恢复

新增 Qwen 专用数据模块、模型适配、两阶段调度及 recipe；复用框架配置、CLI、分布式初始化和可兼容 checkpoint 基础设施。通过 model.family 路由，避免扩散 Z-Image 特定模型逻辑。

FSDP1 按 QwenImage21TransformerBlock wrap，分别包裹 Generator/Fake/Real；开启兼容 FSDP1 的 activation checkpointing。microbatch 默认 1，以保留变分辨率和变参考图布局。所有 rank 的 optimizer 阶段和 collective 次序一致；任何 rank 的非有限损失或异常都应协调失败，不能静默跳过导致不同步。

计数独立持久化：reflow_updates、fake_updates、generator_updates、outer、microbatch/data cursor。恢复同时加载模型、optimizer、RNG、数据快照 hash 和阶段；禁止将3000 Fake updates错误解释成3000 outer。

DMD 每600次 Fake更新保存（outer边界，含对应Generator更新），得到600/1200/1800/2400/3000；REFLOW结束强制保存。checkpoint使用完整性标记，拒绝恢复半写入目录。

按用户最新要求，每20次Fake-score更新输出一次详细debug（fake_updates=20,40,...,3000），在对应outer的Generator更新完成后执行。六步rollout固定使用 `/mnt/hdfs/__MERLIN_USER_DIR__/scripts/complex_prompt.csv` 的全部64条prompt、原尺寸、固定seed和Generator shift_terminal=0.4；另保存DMD真实训练样本上的Fake/Real预测x0及diff。CSV没有参考图字段，因此它验证的是文生图，多参考图编辑由训练debug和单独编辑验收补充。完整产物要求见第8节。全部参与FSDP forward的rank必须按一致顺序执行评测，不能仅rank0调用分片模型。

## 6. 验证门槛与交付

1. CPU单元测试：complete数据导入、缺损拒绝、条件顺序、latent pack/unpack、六点网格、REFLOW loss、epsilon/flow变换、DMD梯度方向、5:1调度、REFLOW GA=1 / DMD GA=4、阶段边界及恢复计数。
2. 固定版本小型Qwen transformer测试：文生图/多图mask、forward/backward、梯度仅流入目标模型，训练适配与官方无cache forward对齐。
3. FSDP1多卡小模型测试：不同rank样本布局、梯度累计、冻结teacher、裁剪、阶段切换、完整checkpoint保存和恢复后续更新。
4. 真实模型短跑：在可用训练资源上完成REFLOW更新与DMD完整outer，检查损失、梯度、参数变化和显存；以测试配置缩短阶段，不能冒充正式1000+3000训练。
5. 六步真实出图：复用固定文本/编辑条件，对照teacher检查latent、图像、参数及明显伪影。
6. 交付：Qwen recipe、不可变数据manifest构建工具、条件缓存工具、两阶段启动命令、resume/eval说明、测试结果与尚未验证的硬件限制。

本次交付已完成数据/条件和scheduler、数学损失与计数、FSDP1 trainer/checkpoint、debug和导出工具，并编写CPU及tiny FSDP1测试。用户要求改完不跑，因此所有运行验收留待后续明确执行。


## 7. 2026-09-21 复核：容易改变训练含义的细节

本节区分“代码已证实”“设计建议”“尚待核对”。新增分析不构成已完成训练验证；只更新本文档。

### 7.1 GA、迭代与实际样本量

最新约定：REFLOW 1000 updates × GA1；DMD 3000 Fake updates + 600 Generator updates，每个 update GA4。

令 D 为数据并行 rank 数、B 为每 rank 的 microbatch 样本数：

| 项目 | 消费的样本次数（含重复采样） |
|---|---:|
| REFLOW 单次更新 | D×B |
| REFLOW 全阶段 | 1000×D×B |
| DMD 单次 Fake 或 Generator 更新 | 4×D×B |
| DMD 一个 outer | 24×D×B |
| DMD 全阶段 | 14400×D×B |
| 两阶段合计 | 15400×D×B |

若以后选用纯 FSDP1 的32卡、B=1，则 REFLOW 有32000次样本消费，DMD有460800次，总计492800次。这不是492800条独立数据，也不是已确定使用32卡。DMD的5个Fake update和1个Generator update各自取新的4个microbatch；如复用数据必须单独记录，不能沿用上述独立取样解释。

REFLOW改成GA1后，不自动按4倍比例放大或缩小LR。1e-4来自Z-Image而非已验证的Qwen学习率，需以Qwen短跑的loss、全局grad norm、参数相对变化和六步样图确认。严格区分“参考配置”和“已验证可训练”。

### 7.2 原始配置实测与最新网格覆盖

以下表格是原始shift_terminal=0.02的历史对照，不是当前训练网格。当前方案按用户指令改为0.4，见表后说明。原始模型配置的CPU实测结果：

| 输出尺寸 | 目标 latent tokens | mu | 六个模型调用 sigma（约值） |
|---|---:|---:|---|
| 2048×2048 | 16384 | 1.312903 | 1, 0.912763, 0.797422, 0.637791, 0.402296, 0.020000 |
| 2400×1696 | 15900 | 1.288508 | 1, 0.911638, 0.795152, 0.634573, 0.399021, 0.020000 |
| 2880×1440 | 16200 | 1.303629 | 1, 0.912338, 0.796564, 0.636573, 0.401054, 0.020000 |

当前Generator网格：2048×2048、shift_terminal=0.4时CPU实测为 `[1.0, 0.94658935, 0.87597263, 0.77823925, 0.63405871, 0.39999998, 0.0]`。前六点各调用模型一次，最后追加的0不调用模型。变换顺序是原始六点→Qwen动态shift→整体拉伸使最后调用点为0.4→追加积分终点0；不是只把原网格最后一个数字替换成0.4。

- 六次模型调用对应六个非零sigma；scheduler额外附加0作为最后积分终点，不能把0当成第七次调用或训练采样点。
- time输入为scheduler timestep/1000；不能重复除1000，不能把Z-Image可能采用的时间方向直接搬过来。
- max_image_seq_len=8192并不代表代码会把mu截到max_shift=0.9。官方calculate_shift是线性外推，16384 tokens实际mu>0.9。必须复现实际行为，不能依据字段名字自行clamp。
- mu来自目标latent token数，不把参考图/VLM token数加入该公式；后者仅影响总注意力长度和显存。
- 每个尺寸使用自己的网格；不得将2048×2048样例作为全部长宽比的常量。
- score的1000点采样网格需独立固定：建议先生成对应尺寸的官方1000步shift网格，再按实际sigma筛选[0.02,0.98]，从合法点均匀采样，并记录实际最小/最大值及采样概率。与“先筛选原始整数t再shift”不同；该顺序是设计建议，不能声称已复现attempt25。

### 7.3 六点覆盖与训练—推理分布差异

现有Z-Image四点权重为[0.8,0.16,0.032,0.008]。若沿同样尾部合并规则扩成六点，会得到[0.8,0.16,0.032,0.0064,0.00128,0.00032]。1000次GA1、单rank独立采样时，最后一点期望只有0.32次；即使32rank独立采样，最后一点全阶段也只有约10.24个样本。若各rank共享采样点，更可能整段训练漏掉该点。

因此首版选择六点均匀采样，记录每点样本数及loss；更新日志记录该次GA累积后的全局grad norm，不冒充每点独立grad norm。分层覆盖可作为后续选择。

REFLOW用固定teacher配对噪声z和x0保持原始耦合；DMD可对数据x0重新采样训练噪声，不能无意中永远复用记录seed。部署评测仍固定seed以便对照。

“六点加噪→去噪”不等于在模型自己的推理中间态上训练：训练输入来自teacher端点直线插值，六步推理的中间态来自Generator前序预测，存在分布偏移。按当前用户意图先保留离散点加噪方案；将完整六步纯噪声评测作为必要门槛。若劣化，需要另行讨论无梯度前缀rollout/backward simulation，不能默认悄悄改成另一训练方法。

还需区分：一次模型调用预测x0=xt-sigma*v，与一次Euler更新到下一sigma是两个操作。六步评测必须做x_next=xt+(sigma_next-sigma)*v；不能每步都直接替换成x0。

### 7.4 Fake loss不能用现有x0默认值冒充epsilon regression

已读源码：当前框架configs/methods/dmd.yaml设置score_loss_target=x0，score_use_weighting=false；现有可选weighting采用sigma/(1-sigma)。当前检出的Mariana m13/dmd.py也包含x0 MSE函数。它们不足以证明用户描述的attempt25 epsilon regression/max_weight/min_alpha公式。

对本项目线性flow约定，设生成样本为y、score噪声为e、score时间为s：

```text
x_s = (1-s)*stopgrad(y) + s*e
v_target = e - stopgrad(y)
x0_hat = x_s - s*v_hat
epsilon_hat = x_s + (1-s)*v_hat
L_epsilon = mean((epsilon_hat-e)^2)
          = (1-s)^2 * mean((v_hat-v_target)^2)
```

这与无权重velocity MSE、无权重x0 MSE不同。单样本固定s时还有：L_epsilon=((1-s)/s)^2*L_x0。该比值在低噪声端很大；加cap以后又是另一个目标。

max_weight=50、min_alpha=1e-4到底用于哪一个比值/分母，必须拿到对应参考函数或显式制定Qwen目标后才能落地，不能猜测alpha是s还是1-s。实现前列出s=0.02、0.5、0.98的loss与梯度数值，分别验证epsilon/flow转换和截断权重。找不到attempt25原函数时，应明确标记为Qwen自定义方案，而非“严格复现attempt25”。

Generator DMD surrogate的可验证定义为：在相同x_s、s和条件上求fake_x0与real_x0，按每张目标图所有latent元素计算denom=mean(abs(stopgrad(y)-real_x0))并以1e-6作下限，g=(fake_x0-real_x0)/denom；使用0.5*mean((y-stopgrad(y-g))^2)*4。符号、0.5因子、每样本归约维度及detach均应有梯度测试。不能把reference tokens计入denom，不能把normalization改成每token归一化。

### 7.5 三个模型的初始化、冻结及CFG语义

Fake从REFLOW Generator复制是本次实施所采用的默认值，配置中明确记录；这是适配选择，非用户指定的Seedream权重要求。REFLOW会改变输出分布，而Fake承担全时间score回归，故该初始化不保证优于原始Qwen；应检查DMD前若干Fake更新的拟合loss与梯度。记录每个模型来源hash，保证Fake与Generator没有参数或optimizer-state别名。

Real必须保持原始Qwen能力；不能在REFLOW→DMD过渡时误把Real覆盖成Generator。Fake在REFLOW不更新；阶段切换只重建该重建的optimizer，不能在resume后重复复制/清空已经训练过的Fake。

Qwen生产脚本的true_cfg_scale=1表示只做conditional forward；不照搬旧框架guidance_scale=0的数字语义。适配层应明确指定conditional-only模式。若以后启用CFG，需要提供负向prompt和保留相同参考图条件的负向embedding；否则Qwen pipeline会禁用CFG，光记录一个大于1的数值不代表实际生效。

DMD计算score差时可用no_grad，因为梯度通过detached surrogate回到Generator；但Generator当前预测必须保留计算图。不要在FSDP1已flatten的模型上按phase反复切requires_grad；用正确的no_grad区域、独立optimizer和一致的forward调度实现冻结。

### 7.6 多参考图条件与数据质量

- Qwen VLM与VAE对alpha的处理不同：VLM分支将RGBA合成到白底，VAE分支保留四通道。不能统一转RGB。
- 官方encode_prompt在当前Transformers版本有读取末层归一化前hidden state的兼容处理；绕过它重新写文本编码会改变条件幅度。固定Diffusers提交、Transformers版本、processor/tokenizer和模板，使用现有已验收链路。
- 参考图顺序、编号、img_shapes、image_pad_mask必须一一对应。loss只覆盖末尾目标latent，而不是把所有拼接图像当监督对象。
- 保存的seed由worker显式传给torch.Generator，文件里的noise才是REFLOW真值。不要仅按seed重新生成并假定跨版本/shape/device必然完全一致。
- 现有生产任务尚未完成，先完成的样本可能偏向文生图、少参考图或较短prompt。建立快照后统计t2i/edit比例、参考图数、尺寸、prompt长度和来源分布；不要把早期完成子集当成原数据全集无偏采样。
- 以(kind,id)标识样本并检查去重。complete.json只是提交标记，仍需核对模型revision、40步参数、payload与参考图可读性；拒绝混入smoke目录或其他模型实验。
- 若从生产样本划出固定评测集，应从训练manifest排除；如果只使用训练样本做可视化debug，必须标注in-sample，不能称作泛化验证。

### 7.7 2048、多图和FSDP1的显存风险

2048×2048目标就有16384 latent tokens，参考图会继续增加序列长度。不要照搬Seedream max_seq_len=32768并通过裁切参考图/prompt/分辨率来绕过；这会改变任务。先测目标长度、参考图长度、VLM文字长度和attention后端的实际峰值，长度超限时显式报错或按资源分桶。

现有engine/fsdp1.py会先module.float()，再以MixedPrecision设置BF16计算。这意味着FP32主参数分片与BF16前向计算，并不等于“全部参数常驻BF16”。还要计入两套AdamW状态、梯度、当前block all-gather峰值、冻结teacher、激活与评测VAE。

FSDP1只分片参数/梯度/optimizer state，不自动分片token激活；现有推理CPU offload成功不能证明三模型训练可装入80GB。activation checkpointing、条件预计算与每rank microbatch=1是起点，需用最重的多参考图样本短跑，不能只测纯文本短序列。

FSDP1.no_sync累积会保留未分片梯度，可能显著增大GA4峰值。默认先考虑每microbatch同步reduce-scatter、loss除以GA，验证累积数学和显存；如采用no_sync优化，必须另外实测。grad clip必须用FSDP感知的全局norm，并在完整GA结束后执行一次。

不同rank可以使用不同样本布局，但必须执行相同数量、顺序的模型forward/backward。无参考图路径不能跳过某些FSDP包裹模块的collective。检查Qwen block包裹、checkpoint重算、动态shape和FSDP1 forward_prefetch的组合；必要时关闭forward_prefetch，不能直接沿用Z-Image的性能开关。

未分片的CPU全模型构造也可能形成多进程host RAM峰值；加载三模型应分阶段并核验峰值，避免32个进程同时各自临时持有三份FP32模型。实际多机通信按训练资源另做NCCL测试，不能拿此前独立单卡推理验收当作跨机通信验收。

### 7.8 调度、验证成本与恢复边界

最新周期为每20次Fake更新（每4 outer）评测64条：3000 Fake更新共150轮、9600次六步样本生成（若加训练前基线则另计64条）；还包含条件处理、VAE解码和保存。这是明显的额外开销，需要单独记录eval wall time，不能用纯optimizer吞吐推算总训练时长。

FSDP1评测不能让只有rank0调用分片Generator，其余rank直接跳过；应共同参与forward，只让相应owner保存样本，或使用经过验证的独立推理权重导出方案。64条不能整除world_size时保证collective对齐，padding样本不得计入指标。

除了三个update计数，checkpoint还需保存当前phase、outer内Fake子步、sampler epoch/cursor、各rank RNG和独立GA配置。优先在optimizer边界保存，不承诺恢复半个GA窗口。3000次Fake更新后还必须完成第600次Generator更新，再写最终完成标记。

训练与验证必须隔离随机数流，评测不能改变后续训练采样轨迹。恢复测试包括REFLOW结束前后、DMD第5次Fake之后/Generator之前，以及最后一个outer。训练data、条件cache及scheduler的hash变化时拒绝“原样续训”，不能静默忽略。

FSDP1 checkpoint必须包含可恢复的optimizer分片，不只导出推理权重；同时提供六步评测所需的Generator导出说明。完成标记在所有rank文件成功且同步后写入；不同world_size恢复若未验证，明确只支持同world_size，不能假定框架DCP调用自动解决全部兼容问题。

### 7.9 实施前需要固定的剩余选择

| 选择 | 当前建议/状态 |
|---|---|
| REFLOW GA | 最新用户允许1；计划默认1，DMD固定4 |
| Generator最后模型调用点 | 已指定sigma=0.4，随后积分到0；score网格不受该覆盖影响 |
| Debug周期与prompt | 已指定每20次Fake更新；使用complex_prompt.csv全部64条，训练Fake/Real x0及diff另存 |
| 六点采样概率 | 首版均匀1/6；分层未实现 |
| REFLOW optimizer | 首版标准AdamW；LR/betas/WD参考Z-Image，不使用其schedule-free optimizer |
| Fake初始化 | 首版固定REFLOW终点独立副本，不与Generator共享storage |
| epsilon loss的cap/min_alpha公式 | 已显式制定Qwen公式，见使用文档；attempt25精确等价仍未核实 |
| score时间shift顺序 | 已实现官方1000点网格后筛选实际sigma，再均匀采样；仍待运行验证 |
| 数据快照大小/评测拆分 | 根据当时有效输出生成清单并统计偏差，不能暗示全量99272已齐 |
| 正式训练卡数/启动时间 | 未指定；不占用或终止正在出图的GPU |

这些选择应先写入最终recipe和run manifest，再开展真实训练；不会因为本次文档分析而自动改变生产任务。

### 7.10 本次分析依据

- 本框架 `configs/recipes/zimage/dmd_refaligned_ode_warmup_fsdp1.yaml`、`configs/methods/dmd.yaml`、`src/verl_distill/algorithms/dmd/method.py`、`src/verl_distill/trainers/dmd.py`、`src/verl_distill/engine/fsdp1.py`。
- 已安装固定提交 `80c7ed262aeffbeb43ef13ae04baeb9b84515a69` 的 Diffusers：`pipelines/qwenimage21/pipeline_qwenimage21.py`、`models/transformers/transformer_qwenimage21.py`、`schedulers/scheduling_flow_match_euler_discrete.py`；模型自己的 `scheduler/scheduler_config.json`。
- `/opt/tiger/mariana/mariana/models/multimodal/gen_image/m13/dmd.py`，仅作为当前检出版本对照，不冒充attempt25的精确源码。
- 本次仅CPU计算了三个尺寸的官方六步网格；没有执行训练或进行GPU资源抢占。


## 8. 最新确定：terminal=0.4 与详细 debug

### 8.1 “采样权重”到底控制什么

六个sigma决定“在哪些噪声程度训练”；六个采样概率决定“每次训练选中哪一个点”。它们是两组不同的量。训练一次只选一个点做加噪→去噪，并不自动把六点各训一次。

如果机械扩展Z-Image几何概率，在1000次、GA1、每rank每microbatch一条样本的期望分配如下（不是实际精确次数）：

| 点序号（从高噪声到低噪声） | 几何概率 | 1000次抽样期望次数 | 均匀概率 | 均匀期望次数 |
|---|---:|---:|---:|---:|
| 1 | 80% | 800 | 1/6 | 约167 |
| 2 | 16% | 160 | 1/6 | 约167 |
| 3 | 3.2% | 32 | 1/6 | 约167 |
| 4 | 0.64% | 6.4 | 1/6 | 约167 |
| 5 | 0.128% | 1.28 | 1/6 | 约167 |
| 6 | 0.032% | 0.32 | 1/6 | 约167 |

现在最后一点为0.4，它仍是六点中噪声最低的一点，但绝不是接近零噪声。将terminal从0.02改为0.4不会自动改变采样概率；若继续采用上述几何权重，第六点仍几乎抽不到。

“均匀采样”是六点每次都有1/6概率，实际次数允许浮动。“分层采样”是例如每6个抽样位置将六个点随机打乱，各覆盖一次，从而控制漏训。两者均不要求每次更新反传六步。多rank实现需定义按rank独立覆盖还是全局覆盖，并保存采样器RNG/cursor；不能为了均匀覆盖破坏collective一致性。

执行计划时首版采用均匀采样，并记录六点各自样本数/loss；分层策略尚未实现。

### 8.2 Debug触发时刻、固定输入和版本记录

- 触发计数严格是DMD阶段fake_updates，不包含REFLOW，不用loader iteration或累计Generator总次数。
- 20是5:1调度的整数倍。在Fake更新20/40/...完成且对应Generator更新4/8/...完成后执行rollout，因此输出反映该outer最新Generator。
- 同一个debug目录同时记录用于诊断的训练forward权重版本（Generator step前、Fake已更新）以及rollout版本（Generator step后），不能将两者标成同一次权重快照。
- 首版默认REFLOW终点输出64条六步基线（debug/fake_step_000000），不在其他REFLOW更新周期出图。
- 固定CSV：`/mnt/hdfs/__MERLIN_USER_DIR__/scripts/complex_prompt.csv`。
- 本次读取确认64行、64个唯一prompt_id；字段为filename/index/prompt_id/width/height/resolution/prompt；全部width/height为32倍数。无参考图列，不擅自给它追加参考图。
- 当前CSV SHA256：`773522a597caa95bc6eadd2f1f32e128755dafa092b343f306f186b963addf2b`。运行启动时保存CSV快照和hash；以prompt_id派生固定seed，不依赖rank分配或评测轮次。后续CSV改变不静默影响已启动任务。

### 8.3 固定64条的六步rollout产物

所有64条每个debug周期均执行6次模型调用，从纯噪声出发，遵循各自尺寸对应的terminal=0.4网格。保存：

- 最终image.png、initial_noise.pt、final_x0_latent.pt、逐样本metadata.json。
- 六步的实际sigma/timestep、下一sigma、预测velocity及预测x0的统计（mean/std/min/max/L2/finite比例）；逐步latent与预测x0张量供复核。
- 逐步x0预览和最终图，生成带step/sigma标注的trajectory/contact sheet；所有图用同一VAE与mean/std解码，不把噪声状态直接当作干净x0解释。
- 原prompt、prompt_id、请求/实际尺寸、seed、模型与scheduler配置hash、Generator checkpoint/update计数、CFG实际是否启用、NFE实际调用次数、耗时和峰值显存。
- 样本级失败记录，汇总完成数/失败数。未全部完成不能写debug成功标记；不能因某条失败而静默漏图。

训练正确性优先：debug在独立RNG上下文运行，不改变之后的训练随机序列；不能复用跨参数更新的KV cache。rollout内部若启用cache，需先验证与无cache六步结果一致。

### 8.4 DMD真实训练样本的Fake/Real x0与diff

每个触发outer，保留该次Generator DMD更新的4个GA microbatch、各数据并行rank上的实际诊断张量。禁止为了凑debug重新抽一个score timestep而伪称它是刚才loss所用输入。数值快照detach后存CPU，不额外保留训练计算图。

必须使用相同生成样本y、相同score噪声e、相同x_s、相同s、相同文本/参考图条件比较两个score分支：

```text
fake_x0 = x_s - s * v_fake
real_x0 = x_s - s * v_real
diff_x0 = fake_x0 - real_x0
normalized_dmd_direction = diff_x0 / clamp(mean(abs(y-real_x0)), min=1e-6)
```

保存目标区域的：

- y（Generator预测x0）、score_noise、score_noisy_latent、v_fake、v_real、fake_x0、real_x0、diff_x0、DMD normalization denominator及normalized direction，均含原始shape/dtype。
- Generator输入的六点索引/sigma、score sigma、sample kind/id、prompt、reference路径/顺序/内容hash、width/height、GA索引、rank、三种update计数与optimizer LR。
- 同一VAE解码的generator_x0.png、fake_x0.png、real_x0.png，及并排对照图。编辑样本附参考图缩略图，保留顺序。
- signed latent diff与absolute latent diff原始张量、统计和空间热图（例如对channel取mean(abs(diff))），标记色标与尺度。跨周期尽量固定可视化尺度，记录任何截断范围，避免各图独立拉伸造成误读。
- 解码后pixel差值图：decode(fake_x0)-decode(real_x0)，保存原始浮点差值和可视化。**不得把decode(fake_x0-real_x0)当作两张图的差值**，因为VAE非线性且含latent反归一化。
- diff的MAE/RMSE/L2/max、Fake/Real相对y的误差、cosine、denominator、loss未加权/加权值和nonfinite计数。明确按样本归约，再做DP聚合，不混入参考图tokens。

在对应outer的第5次Fake更新窗口额外记录4个GA microbatch的epsilon_target/epsilon_pred、Fake loss的权重及截断前后值；这些与随后Generator窗口是不同样本，分别命名，不拿两批的结果直接相减。

### 8.5 每次更新的标量日志与存储布局

详细标量日志每个optimizer update写入，不能只每20次才记录：phase、reflow/fake/generator/outer计数、GA、有效batch、各六点命中数、score sigma分布、各项loss、Fake loss权重与cap命中率、DMD denominator、梯度裁剪前/后全局norm、LR、参数变化探针、step/data/forward/backward耗时、峰值显存、非有限值与样本异常。

建议布局：

```text
OUTPUT/debug/fake_step_000020/
  event.json
  prompts_snapshot.csv
  rollout/<prompt_id>/
    image.png
    initial_noise.pt
    final_x0_latent.pt
    trajectory.png
    steps/...
    metadata.json
  train_dmd/rank_000/ga_00/<sample_id>/
    generator_x0.pt
    fake_x0.pt
    real_x0.pt
    diff_x0.pt
    normalized_direction.pt
    generator_x0.png
    fake_x0.png
    real_x0.png
    latent_diff_heatmap.png
    pixel_diff.pt
    comparison.png
    metadata.json
  train_fake/rank_000/ga_00/<sample_id>/...
  summary.json
  COMPLETE
```

不同rank、GA、样本不得写同一个路径。COMPLETE只在所有预期rollout和训练诊断文件写完后出现。恢复时避免同一fake_updates覆盖旧事件；记录对应checkpoint/run-id和是否为重试。

每20Fake更新、3000Fake更新共150个debug周期，仅固定prompt就有9600条六步rollout；还要保存训练4GA×DP rank诊断和中间张量，存储与VAE解码开销可能较大。正式启动前估算字节数、磁盘容量和I/O耗时；默认不因性能原因减少已约定的debug覆盖。如需抽样、降低频率或丢弃张量，必须先明确调整要求。

## 9. 实施交付（2026-09-21）

- 新增Qwen数据快照/条件缓存、模型adapter和官方scheduler适配、REFLOW/DMD数学及阶段计数、FSDP1 trainer、原子checkpoint与同world-size恢复、全量debug、Generator导出。
- 新增recipe和显式启动脚本；框架CLI通过model.family=qwen_image21路由。未改原始生产scheduler及生产文件。
- 新增CPU/小Qwen block测试与手动tiny FSDP1恢复测试；按用户“改完后不需要跑”，未执行这些测试，也未运行训练/推理/缓存。
- 本次只进行源码审阅与语法静态检查；不能据此断言实际A100显存、跨机通信、模型适配或恢复正确性已经通过验收。
- 正式数据manifest、条件cache、训练环境和启动资源需在后续运行阶段准备，未声称当前已生成。

实现默认值、完整公式、操作命令、产物路径与资源限制见 [Qwen使用文档](../qwen_image21.md)。
