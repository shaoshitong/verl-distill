# Qwen-Image-2.1：当前复现入口

新增配方 [`reflow100_lr1e5_ga4_dmd_fsdp1.yaml`](../configs/recipes/qwen_image21/reflow100_lr1e5_ga4_dmd_fsdp1.yaml)：从原始 HF 开始，REFLOW 100 次更新、LR=1e-5、GA=4，之后 DMD Fake 3000 / Generator 600 次更新，5:1、GA=4。启用两阶段权重 offload，debug 为每卡一条、共 8 条的 6/25 步对照。

环境、数据预处理和启动命令见 [README](../README.md#reflow--dmd-for-qwen-image-21)。以下保留历史配方和验收记录；其中 1000 步、LR=1e-4、GA=1、64 条 debug 等描述只适用于对应的旧版本。新实验以所选 YAML 为准。

## 下次启动默认的性能配置（2026-09-21）

当前运行不热加载、不自动重启。`configs/recipes/qwen_image21/reflow_dmd_fsdp1.yaml` 下次默认启用：

```yaml
runtime:
  data_ordering: bucketed_v1
  bucket_batches: 64
  attention_backend: flash2_segmented
  debug_prompt_count: 8
  debug_comparison_steps: 25
```

### 数据分桶

每512条（8卡×64个batch）随机样本按估算token数排序、每8条成组，再随机打乱组顺序和组内rank分配。估算量为目标latent token数 + 每张参考图16384 tokens + prompt字符数/3，属于不读取payload的成本代理，并非精确的VLM分词或FLOPs。所有原始样本和原有epoch末尾padding规则保留，不丢弃多图任务。固定seed可重现，每个epoch重新排列。

从旧random checkpoint切换时，需要明确开启下面的迁移选项；只重新分桶当前epoch未消费的全局样本后缀，已消费部分保持原样，不重置optimizer、阶段计数或训练RNG。cursor记录分桶边界、成本哈希与pool大小，再次恢复可重现顺序。只放行data_ordering、bucket_batches、attention_backend三项合同差异，数据/模型/LR/GA等仍严格验证。更改已有bucketed checkpoint的成本或pool大小会拒绝。

### Attention

使用仓库内`QwenImage21Flash2Processor`，不修改site-packages。保留官方QKV、RoPE、分段边界、输出投影和缓存语义；图像段使用全prefix attention，文本段使用FlashAttention2的右下对齐因果mask。Q长度=end-start，K长度=end，条件`key_index <= query_index + start`与官方拼接mask相同。无需torch.compile，不引入可变分辨率重编译开销。带显式padding mask或CPU/FP32调用回退官方SDPA。

需要可导入的`flash-attn>=2.1`，本机验证版本2.8.3.post1；新节点按匹配的PyTorch/CUDA构建安装，例如`python -m pip install flash-attn==2.8.3.post1 --no-build-isolation`。缺依赖在模型启动配置时直接报错，不静默声称启用成功。原始SDPA可设`runtime.attention_backend: sdpa`作为对照。Generator、Fake、Real都使用所选处理器，训练目标/时间步/CFG不变；融合内核会产生BF16舍入差异。

### 启动与续训

沿用原来的环境、数据和条件缓存，启动脚本仍是`scripts/train_qwen_image21.sh`。全新训练不需要迁移开关。从当前旧训练checkpoint启动优化版本时：

```bash
export QWEN21_RESUME_FROM=/absolute/path/to/checkpoints/reflow_step_000500
export QWEN21_ALLOW_INFRA_RESUME_CHANGE=true
export OUTPUT_DIR=/absolute/path/to/a_fresh_run
NPROC_PER_NODE=8 bash scripts/train_qwen_image21.sh
```

这只是手动启动说明，不会杀掉当前训练，也没有自动切换监控。是否采用这些新设置记录在新run配置和infra审计中；若旧checkpoint发生infra迁移，额外写`infra_migration.json`。

### 已完成验证和限制

- 31项CPU回归测试：采样覆盖/跨epoch、旧checkpoint后缀迁移、迁移后再次恢复、合同限制、调度器、debug写盘及数学检查。
- CUDA小型真实Qwen：0/1/2/5张参考图、交错文本，前向和梯度对照通过；输入梯度最大相对差约0.14%，参数梯度最大相对差约0.12%。padding回退与官方输出一致。
- 两卡小型Qwen FSDP1：Flash2 + gradient checkpointing、GA4、阶段梯度隔离、checkpoint恢复重放通过。
- 两卡debug编排：8条6/25步输出、GA快照、RNG恢复、单rank解码/写盘失败传播通过（使用模拟解码器，不是生产VAE性能测试）。
- 实际5662条样本的离线分桶模拟：平均batch最大估算token数37954→23697（约下降37.6%），仍覆盖全部5662条，含epoch padding共5664次采样。这不是实测训练加速比例。
- 未重启7B生产训练，尚未测完整新配置的端到端速度。验收证据位于`verl-distill-runs/qwen21_infra_optimized_audit_20260921/`。

## 当前 debug 设置（8 条，6 / 25 步，无 CFG）

新启动进程使用 `runtime.debug_prompt_count: 8` 和 `debug_comparison_steps: 25`。
评测 manifest 仍保留原始 64 条及哈希，但实际按原顺序只取前 8 条；8 卡时 rank 0–7 各负责一条。每 20 次 REFLOW / Fake 更新的触发规则不变。

- `debug/<event>/rollout/<id>/`：6 步，训练使用的动态 shift 和末次调用 sigma=0.4；保留逐步张量、预览与最终图。
- `debug/<event>/rollout_25step/<id>/`：25 步，原模型 scheduler 配置（动态 shift，shift_terminal=0.02），仅正条件、CFG=1；只保存 initial_noise.pt、final_x0_latent.pt、image.png、metadata.json。metadata 记录全部时间步和原 scheduler 配置。
- 两组使用相同 seed、尺寸、条件和完全相同的初始噪声；25 步恰好 25 次模型前向，不执行负条件分支。
- DMD 的实际训练张量快照仍覆盖各 rank 的所有 GA；新限制作用于固定 prompt 评测。

REFLOW 从数据集读取配对的 epsilon（initial noise）和 x0，计算 `xt=t*epsilon+(1-t)*x0`，loss 为 `mean((student(xt,t,condition)-(epsilon-x0))**2)`。t 均匀选自当前尺寸对应的六个实际 sigma，不做 1-t 反转。此公式仅描述 REFLOW。

这两个 debug 参数不纳入训练数值合同，允许已有 checkpoint 保持数据、优化器、模型及 RNG 身份续训。代码不会热加载；运行中的旧进程仍使用 64 条六步评测，切换需从已提交 checkpoint 启动新输出目录。以下历史验收记录中的 64 条描述是当时配置。

# Qwen-Image-2.1 REFLOW → DMD

2026-09-21 启动验收：用户已授权停止本节点推理并启动训练。19 项单元测试、双卡和 8 卡小模型 FSDP1/GA/冻结参数/checkpoint 恢复测试通过；5,662 条训练样本已完成哈希校验。条件缓存已发布，实际 7B 模型已完成至少 6 次 REFLOW 更新，约 22 秒/次，尚未进入完整 DMD 阶段。实际进度见文末运行目录；小模型测试不代表真实 7B 训练或出图质量已经验收。

设计依据：[接入计划及注意事项](plans/qwen_image21_reflow_dmd.md)。入口配置：[reflow_dmd_fsdp1.yaml](../configs/recipes/qwen_image21/reflow_dmd_fsdp1.yaml)。

## 训练含义

| 项目 | 实现 |
|---|---|
| REFLOW | 1000 次 Generator 更新，默认 GA=1，六点均匀采样 |
| DMD | 3000 次 Fake 更新，5 Fake → 1 Generator，共 600 次 Generator 更新；二者 GA=4 |
| 单卡 microbatch | 1，保留原尺寸和全部参考图 |
| 初始化 | Generator 从原始 Qwen 开始 REFLOW；进入 DMD 时 Fake 复制 REFLOW 终点为独立参数；Real 始终为冻结的原始 Qwen |
| REFLOW optimizer | 标准 AdamW，LR=1e-4，betas=(0.9,0.999)，WD=0.01，clip=1 |
| DMD optimizer | Gen：5e-7/(0.9,0.999)；Fake：5e-6/(0.9,0.95)；WD=0.1，clip=1 |
| 并行/精度 | FSDP1，按 Qwen block 分片；FP32 主参数/梯度/reduction，BF16 forward，FP32 loss；activation checkpointing |
| Guidance | conditional-only，与生产 Qwen 的 true_cfg_scale=1 对应 |
| 不启用的分支 | LR scheduler、warmup、EMA、GAN、多层 feature loss、跨更新 KV cache |

一次训练 microbatch 只选六点之一：对数据端点加噪，调用一次 Generator 预测 x0。REFLOW 使用数据保存的真实 initial_noise；DMD 为数据 x0 重新采样噪声。六步串行 Euler rollout 用于评测，不在每次训练更新上反传六步。

六点各以 1/6 概率独立抽取。例如 1000 次单 rank 抽样，各点期望约 167 次，并非严格每点 167 次。多 rank 独立采样、统一执行相同阶段；详细日志记录各点实际命中和 loss。

Generator 沿用 Qwen 按目标尺寸计算动态 shift，再将 `shift_terminal` 覆盖为 0.4；2048² 的 sigma 约为：

```text
调用：1, 0.94658935, 0.87597263, 0.77823925, 0.63405871, 0.39999998
积分终点：0（不再调用模型）
```

每步 `x_next=x+(sigma_next-sigma)*velocity`，最后一步从 0.4 积分到 0。其他尺寸由官方 scheduler 重新计算。Score 使用独立的原始 Qwen scheduler（terminal=0.02）：先生成官方 1000 点动态 shift 网格，再从实际 sigma 落在 [0.02,0.98] 的点均匀取样。允许 1e-7 的浮点边界误差后 clamp，不把 score 改成六点或 terminal=0.4。

## 损失定义与参考配置的区别

以下均只作用于目标图 latent，不含参考图 tokens。数据已是 diffusion-normalized latent，训练时不再重复归一化。

```text
REFLOW:
  x_s = (1-s)*x0 + s*saved_noise
  loss = mean((v - (saved_noise-x0))²)

Fake:
  x_s = (1-s)*stopgrad(y) + s*noise
  fake_x0 = x_s - s*v_fake
  epsilon_hat = x_s + (1-s)*v_fake
  weight_raw = ((1-s)/max(s, 1e-4))²
  weight = min(weight_raw, 50)
  loss = mean_per_sample(weight * MSE(fake_x0, stopgrad(y))) * 1

Generator:
  real_x0 = x_s - s*v_real
  denominator = max(mean_per_sample(abs(stopgrad(y)-real_x0)), 1e-6)
  direction = (fake_x0-real_x0)/denominator
  loss = 0.5*mean((y-stopgrad(y-direction))²) * 4
```

Fake 的公式是**显式制定的 Qwen 目标**：不截断时等价于 epsilon MSE；cap 后会减弱低 sigma 点的 epsilon 回归权重。`min_alpha` 在本实现明确指 noise coefficient `s` 的分母下限。s=0.02/0.5/0.98 时 raw weight 约为 2401/1/0.00041649，cap 后为 50/1/0.00041649。尚未取得 attempt25 的原始权重函数，因此不宣称精确复现其 loss。均匀采样、Fake 初始化和标准 REFLOW AdamW 是本次落地的明确默认值。

## 环境

单独使用训练环境，避免升级当前生产出图的 venv。固定 Python 3.12、torch 2.8.0+cu128、torchvision 0.23.0+cu128、Transformers 5.17.0、Diffusers Git commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`。模型 revision 为 `b3179ad355be050328e483a9dfdd9e60cd62adfa`。训练/条件缓存入口检查包版本和 Diffusers Git 来源。

后续在每个训练节点按需安装（本次未执行）：

```bash
cd /mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill
python3.12 -m venv /tmp/qwen-image21-train-venv
/tmp/qwen-image21-train-venv/bin/python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
/tmp/qwen-image21-train-venv/bin/python -m pip install -e '.[qwen21,dev]'
```

Qwen 的 `qwen21` extra 和 Z-Image 的旧 `train` extra 使用不同 Diffusers 版本，不在同一环境合并安装。本地节点 `/tmp` 目录不共享；各节点环境和参考图解压目录分别准备。模型目录需完整包含 transformer、VAE、text_encoder、processor、scheduler；启动按实际文件内容校验模型身份，哈希扫描会增加启动 I/O。

## 数据快照与条件缓存

新建独立目录，只读取生产输出。训练 manifest 固定当前有效 complete.json 集合，不自动纳入后来完成的样本。默认核验全部 payload SHA256；可选 `--fast-index` 只跳过建索引阶段 payload hash，训练首次读取仍验证。参考图内容、模型 revision、40-step 生产参数、维度、唯一性均核验；拒绝样本及分布统计记录在 summary.json。

固定评测 CSV 的全部 64 条 prompt 按原尺寸进入 eval.json；相同 prompt 文本从 train.json 排除。CSV 没有参考图，编辑诊断使用实际训练样本。

```bash
export QWEN21_REPO=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill
export QWEN21_ROOT=/mnt/hdfs/__MERLIN_USER_DIR__/Qwen-Image-2.1-test
export QWEN21_PYTHON=/tmp/qwen-image21-train-venv/bin/python
export PYTHONPATH="$QWEN21_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export QWEN21_MODEL_PATH="$QWEN21_ROOT/model"
export QWEN21_REFERENCE_ROOT=/tmp/qwen21-dataset
export QWEN21_SNAPSHOT="$QWEN21_ROOT/training_snapshots/reflow_dmd_v1"
export QWEN21_CONDITION_CACHE="$QWEN21_ROOT/training_conditions/reflow_dmd_v1"
export QWEN21_DEBUG_CSV=/mnt/hdfs/__MERLIN_USER_DIR__/scripts/complex_prompt.csv

"$QWEN21_PYTHON" -m verl_distill.tools.prepare_qwen_image21 snapshot \
  --output-root "$QWEN21_ROOT/dataset_inference_20260921" \
  --reference-root "$QWEN21_REFERENCE_ROOT" \
  --prompts-csv "$QWEN21_DEBUG_CSV" --snapshot-dir "$QWEN21_SNAPSHOT"

export QWEN21_TRAIN_MANIFEST="$QWEN21_SNAPSHOT/train.json"
export QWEN21_EVAL_MANIFEST="$QWEN21_SNAPSHOT/eval.json"
```

条件缓存复用官方 RGBA/VLM/VAE 处理、参考图顺序、image mask 和 img_shapes；每条缓存按模型内容、prompt、参考图内容及预处理语义绑定。下面是一进程构建示例，需要可用 GPU；不与生产进程争用资源：

```bash
"$QWEN21_PYTHON" -m verl_distill.tools.prepare_qwen_image21 cache \
  --manifest "$QWEN21_TRAIN_MANIFEST" --eval-manifest "$QWEN21_EVAL_MANIFEST" \
  --cache-dir "$QWEN21_CONDITION_CACHE" --model "$QWEN21_MODEL_PATH" \
  --reference-root "$QWEN21_REFERENCE_ROOT" --num-shards 1 --shard-id 0 --device cuda:0
"$QWEN21_PYTHON" -m verl_distill.tools.prepare_qwen_image21 publish \
  --manifest "$QWEN21_TRAIN_MANIFEST" --eval-manifest "$QWEN21_EVAL_MANIFEST" \
  --cache-dir "$QWEN21_CONDITION_CACHE" --num-shards 1
```

可由多个独立缓存进程分片：每个进程使用相同 `--num-shards K`，唯一 `--shard-id 0..K-1` 和明确的 GPU；全部完成后用 K 发布。工具检查分片身份、覆盖量和文件 hash，生成 index.json 后禁止覆写。训练只读已发布缓存，VLM 不常驻训练 GPU；VAE 仅在 debug 加载到 GPU，结束后回 CPU。

manifest 是逻辑快照，不复制全部生产文件。源 payload 和已发布缓存均需保持不可变。可通过 `QWEN21_DATA_ROOT` 指向同内容的本地数据副本，通过 `QWEN21_REFERENCE_ROOT` 指向本节点参考图副本。

## 后续启动与恢复

设置上述变量后，单机显式启动：

```bash
export OUTPUT_DIR=/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/qwen21_v1
export NPROC_PER_NODE=8
bash "$QWEN21_REPO/scripts/train_qwen_image21.sh"
```

四节点 32 卡仅为启动方式示例，未验收显存和跨机通信。每个节点设置相同共享变量和 rendezvous，分别执行一次启动脚本：

| NODE_RANK | 节点 IPv6 |
|---:|---|
| 0 | fdbd:dc03:16:80::198 |
| 1 | fdbd:dc03:16:84::210 |
| 2 | fdbd:dc03:16:88::202 |
| 3 | fdbd:dc03:16:88::210 |

```bash
export NNODES=4 NPROC_PER_NODE=8
export NODE_RANK=0  # 各节点分别为 0/1/2/3
export RDZV_ENDPOINT='[fdbd:dc03:16:80::198]:29641'
export RDZV_ID=qwen21_reflow_dmd_v1
bash "$QWEN21_REPO/scripts/train_qwen_image21.sh"
```

SSH 的 5000 端口不是 torchrun rendezvous 端口。`NEW_CLUSTER_SETUP.md` 的原推理方式是独立单卡任务，没有验证 NCCL。脚本不写死 H100 的网卡/HCA/IB 设置；A100 的 `NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、IB 路径及端口连通需按实际训练作业配置。所有节点必须使用相同代码、模型、manifest、cache 和共享 OUTPUT_DIR。当前不会自动 SSH、占卡、停止任务或启动训练。

只解析配置时，可直接调用 CLI 的 `--dry-run`，不要经 torchrun 启动脚本调用：

```bash
"$QWEN21_PYTHON" -m verl_distill.cli.train \
  --config "$QWEN21_REPO/configs/recipes/qwen_image21/reflow_dmd_fsdp1.yaml" --dry-run
```

新启动进程在 REFLOW 第 **1、500、1000** 次更新后保存 `checkpoints/reflow_step_XXXXXX/`；REFLOW 每 **20** 次更新执行 64 条六步 rollout，保存与 debug 同步触发时先保存，再构造 VAE 并执行 debug；DMD 在 Fake 600/1200/1800/2400/3000、完成对应 Generator 更新后，也先保存再 debug。DCP 含模型/optimizer 分片，各 rank 保存 RNG 和数据 cursor，state.json 保存阶段、计数及合同 hash。仅提交完成的目录可恢复；拒绝 `.incomplete`。新 checkpoint 的 COMPLETE 还记录 state.json 哈希，state.json 记录全部分片/sidecar 文件大小，恢复前拒绝缺失、截断文件和被修改的 state.json。文件大小检查不是全部 tensor 内容的校验和。

恢复仍调用同一脚本，设置 `QWEN21_RESUME_FROM=/.../checkpoints/fake_step_000600`，并选择**新的 OUTPUT_DIR**；其余数值配置和数据/模型身份保持一致。当前只支持相同 world size，不恢复半个 GA 窗口，也不重用旧 debug 目录。REFLOW 终点恢复只执行一次 Fake 初始化；DMD 恢复不重新复制 Fake 或清空 optimizer。REFLOW 中途可从第 1 或 500 步 checkpoint 恢复；中途故障需重做最后一个已保存 checkpoint 之后的更新。

## 输出位置

```text
OUTPUT_DIR/
  run.json
  logs/rank-00000.jsonl                 # 每个 optimizer update；各 rank 独立文件
  checkpoints/reflow_step_001000/...
  checkpoints/fake_step_000600/...
  debug/fake_step_000000/               # REFLOW 终点 64 条基线
  debug/fake_step_000020/               # Fake=20，Generator=4 更新后评测
    event.json
    prompts_snapshot.csv
    rollout/<prompt_id>/
      image.png
      initial_noise.pt
      final_x0_latent.pt
      trajectory.png
      steps/00/{latent.pt,velocity.pt,predicted_x0.pt,predicted_x0.png}
      metadata.json
    train_dmd/rank_000/ga_00/<sample_id>/
      generator_x0.pt / fake_x0.pt / real_x0.pt / diff_x0.pt / abs_diff_x0.pt
      score_noise.pt / score_noisy_latent.pt / fake_velocity.pt / real_velocity.pt
      denominator.pt / normalized_direction.pt
      generator_x0.png / fake_x0.png / real_x0.png / comparison.png
      latent_diff_heatmap.png / pixel_diff.pt / pixel_diff_heatmap.png
      metadata.json
    train_fake/rank_000/ga_00/<sample_id>/...  # 第5次 Fake 更新窗口，另含 epsilon 诊断
    summary.json
    COMPLETE
  TRAINING_COMPLETE.json                # 精确完成 1000/3000/600 后才写入
```

训练快照含触发窗口的全部 4 GA、全部 rank；DMD x0 来自刚才 loss 的相同 noisy latent、score 时间和条件。训练快照为 optimizer 更新前的 forward，64 条 rollout 为对应 Generator 更新后的权重，元数据分别记录版本。Fake 窗口与 Generator 窗口使用不同训练样本，分别保存。

`diff_x0=fake_x0-real_x0` 保留 signed/absolute 原始 latent；pixel diff 是两个 x0 **分别 VAE 解码后相减**。热图使用固定量程：latent mean-abs 1.0、raw pixel mean-abs 2.0；记录截断比例和真实统计，不把 decode(diff) 当作图像差。

各步日志含六点命中/每点 loss、样本尺寸/参考图数/条件长度、loss 权重与 cap、denominator、x0/diff 数值统计、全局 clip 前 norm、clip 后 norm 上界、LR、参数分片变化探针、耗时和显存。clip 后字段明确是上界，不冒充二次实测 norm。所有 debug 操作保存/恢复 RNG；未完成全部 64 条和训练快照时不写 COMPLETE，失败会终止训练而非静默漏图。

DMD 共 150 次 debug，固定 prompt 9600 条六步结果；REFLOW 每 20 步一次，共 50 次 / 3200 条六步结果。2048² 的单个 FP32 latent 约 4 MiB，每条 rollout 的 6 步×3 张量加首尾约 80 MiB，仅 rollout 张量即约 1000 GiB；另加所有 rank 的训练诊断、PNG、条件缓存及 checkpoint。当前默认全量保存，实际总量会更大，需按数据尺寸和参考图规模预留存储。

## 权重导出

使用与 checkpoint 相同的 world size，显式 torchrun 调用 `verl_distill.tools.export_qwen_image21`。单机示例：

```bash
"$QWEN21_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m verl_distill.tools.export_qwen_image21 \
  --checkpoint "$OUTPUT_DIR/checkpoints/fake_step_003000" \
  --model "$QWEN21_MODEL_PATH" --output /path/to/new/exported_transformer
```

四机 checkpoint 用对应四机 rendezvous 参数在所有节点执行。导出完整 Generator 为 Diffusers transformer config + 分片 safetensors，保留 FP32 主权重，rank0 需容纳全模型 CPU state。导出目录不是完整 pipeline：推理仍需原始 processor/text_encoder/VAE，单独载入导出的 transformer，并在独立 scheduler 配置设置 `shift_terminal=0.4`、6 steps、conditional-only。distillation.json 记录这些语义；它不会被 Diffusers 自动应用，不能直接沿用原始 0.02 scheduler。训练内的六步 debug 已使用正确的 0.4 网格。

## 验收范围

已执行的启动检查见下文。真实 7B REFLOW 已在 8 × A100 上运行并覆盖多参考图；真实三模型 DMD、正式 64 prompt GPU 解码、真实 7B checkpoint 的落盘和重启恢复仍需实际验收。小模型测试不能替代上述运行规模的显存、存储吞吐和质量验证。

## 2026-09-21 启动检查与冻结策略

本节点启动目录：`/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/qwen21_reflow_dmd_fsdp1_20260921`。
`launch.sh` 先在 8 卡分片生成条件缓存，再发布索引并启动 8 卡训练；正常结束后启动占卡程序。
独立训练环境 `/tmp/qwen-image21-train-venv` 复制当前已经验证的 Qwen Python 包，并只读复用原环境的 Torch 依赖；没有升级推理环境。

Generator 和 Fake 均冻结 `time_text_embed.*` 和 `txt_in.*`；训练 `transformer_blocks.*`、`img_in.*`、`modulation.*`、`norm_out.*`、`proj_out.*`。后面三项是 DiT 的调制和输出层，并非外部条件编码器。此模型没有独立的 CFG embedding，配方也不传 CFG 条件。Real、VLM 和 VAE 全部冻结。

FSDP1 使用 `use_orig_params=True` 以支持根 wrapper 内冻结和可训练参数共存；每个 QwenImage21TransformerBlock 单独包裹，并断言包裹数与模型 block 数一致。优化器只接收 requires_grad 参数。训练模型启用非 reentrant gradient checkpointing。每个 rank 写入 `infra-rank-XXXXX.json` 记录真实参数数目、冻结名称、包裹数和 checkpointing 状态。冻结策略也纳入续训 contract，禁止静默换策略恢复。

官方 pipeline 向 transformer 传入 `scheduler timestep / 1000`，也就是 sigma；Qwen 内部 time embedding 再乘 1000。这里直接传 sigma，**不做 `1-sigma` 反转**。输出为 velocity，`x0 = x_sigma - sigma * velocity`。

已执行：19 项单元测试通过；双 GPU 和完整 8 GPU 小 Qwen 的 FSDP1、GA4、更新隔离、冻结参数保持不变、checkpoint 保存和恢复重放通过。实际 7B 大模型运行结论以运行目录日志为准，以上小模型测试不代表完整训练已经跑通。


## 2026-09-21 checkpoint / debug 安全修复

- 保存节点先完成 checkpoint，再调用 debug（包括 VAE 构造）；保存失败不进入 debug。
- Qwen 仅在本 rank 的 GPU 上采集/恢复 RNG；不再用 get_rng_state_all 初始化其他可见 GPU 的上下文。
- 模型和 AdamW 由 DCP 分片保存到 `.incomplete`；cursor/RNG 在各 rank 本地 flush/fsync 后共同确认。任一 rank 的 sidecar 写入失败，全部 rank 报错，目录不提交。
- 所有文件成功后写 state.json / COMPLETE，再重命名为最终目录；恢复前核对元数据摘要和文件大小。保存时将 state_dict 分片复制到 CPU 写盘，这不同于训练期间开启 CPU offload。
- 即使 debug 清理 VAE 报错，也通过 finally 恢复模型 train/eval 状态及 RNG。没有把失败的 debug 标成成功，也不在损坏的通信组上盲目尝试再次保存。
- 保存频率未更改：REFLOW 1000；DMD 每 600 Fake 更新。非保存节点的 debug 失败只能从上一个已保存 checkpoint 恢复。
- checkpoint 不保存尚未写出的 debug 临时张量。恢复训练不自动补跑失败的 REFLOW debug，也无法仅靠更新后的 checkpoint 重建更新前的精确训练诊断张量。

**生效范围：以上是源码修复，已经运行的 `qwen21_reflow_dmd_fsdp1_20260921` 进程不会热加载这些改动，仍按旧顺序运行。没有为了应用修复而停止它；当前 REFLOW 尚无 checkpoint，重启将丢失已有更新。**

审计输出：`/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/qwen21_checkpoint_audit_20260921`。
新进程恢复测试：先用 `tests/fsdp1_qwen_image21_smoke.py --output <共享测试目录> --fresh-process save` 保存并计算下一步对照；退出后另起同 world size 的 torchrun，改用 `--fresh-process restore`。测试新建模型及空 AdamW 后恢复，比较下一步参数、AdamW 状态和 cursor。
故障注入单测：`tests/unit/test_qwen_checkpoint_safety.py`。
分布式 debug/IO 测试：`tests/fsdp1_qwen_image21_debug_smoke.py --output <全新共享目录>`，使用 tiny Qwen 和显式模拟解码器；该测试不代表真实 2048 GPU 解码已通过。

审计实测结果：28 项单元/回归测试通过；共享目录的新进程恢复比对、两轮完整 64 条六步 debug、单 rank 解码和写盘故障注入均通过。真实 VAE 32×32 与 2048×2048 CPU 解码/PNG 写入通过；仍未声称真实 7B + 2048 GPU debug / 大 checkpoint 已验收。完整证据见上述审计目录的 AUDIT.md。`launch_fixed.sh` 是已准备但未执行的修复版新运行入口，不会自行停止当前进程。


## 2026-09-21 用户授权重跑与新频率

修复版运行目录：`/mnt/hdfs/__MERLIN_USER_DIR__/Z_image_RL_DMD/verl-distill-runs/qwen21_reflow_dmd_fsdp1_ckptfix_20260921`。
本次从原始权重重新开始 REFLOW，复用已验证的 5,662 条数据快照与 5,726 份条件缓存。
REFLOW checkpoint 第 1、500、1000 步；64 条六步 debug 每 20 步，目录分别为 `debug/reflow_step_000020/`、`reflow_step_000040/` 等，避免 fake_updates=0 导致覆盖/目录冲突。第 500、1000 步先提交 checkpoint，再出图，且只出一次。
DMD 仍按 Fake 更新计数，每 20 次 Fake 更新出图，每 600 次 Fake 更新先存 checkpoint。正常结束自动运行占卡脚本。
上节“当前旧进程未热加载”的说明描述修复前的旧运行；本次用户已授权停止旧进程并重跑，新进程直接加载修复版源码。

第 1 步真实 7B checkpoint 已落盘（约 79.1 GiB），CPU 逐块读回 7,341 个 tensor chunk 和 8 个 rank sidecar 全部通过，形状/dtype 正确且全有限。报告位于修复版运行目录 `preparation/checkpoint_step1_readback.json`；这是全量文件读回验证，不冒充完整 7B 分布式重启重放。

修复版第 20 步已完成真实 7B + GPU VAE 的 64 条六步 debug（331.84 秒），1,792 个文件 SHA256/PNG 校验通过，并已返回训练继续第 21、22 步。8 个 rank 的第 21 步数据和时间点采样序列与无中间 debug 的旧运行一致。新运行当前验证记录以 RUN_NOTES.md 为准；真实三模型 DMD 和完整 7B 重启重放仍未执行。

### Generator backward 的 score 权重卸载（2026-09-22）

配置 `runtime.offload_scores_for_generator_backward: true` 可在每个 Generator microbatch 完成 Fake/Real 的 no-grad 前向后，将这两个 FSDP1 模型的 FP32 本地权重分片移到 CPU，完成 Generator backward 后再移回 GPU。Fake 阶段保持原有 GPU 前向、反向和 AdamW 更新；优化器参数对象及 GPU Adam 状态不变。GA、损失公式和训练时间采样均不改变。

实现位于 `engine/qwen_score_offload.py`，同步修改 flat parameter、`_local_shard` 和原参数 views，不能用普通 `model.cpu()` 替代。该实现依赖当前固定的 PyTorch 2.8 FSDP1 私有接口，升级版本必须重新验证。开启此项恢复旧 checkpoint 时需要 `allow_infra_resume_change: true`，仅放行基础设施配置差异，不放行优化器或损失参数差异。

日志每条 Generator microbatch 的 `score_offload` 包含 `weight_bytes`、`allocated_bytes_freed`、`offload_seconds` 和 `reload_seconds`。8 卡实测每卡释放约 6.63 GiB；GPU/CPU 往返约 4.8 秒/microbatch，随机器带宽变化。

两卡验证可先运行 `tests/fsdp1_qwen_image21_smoke.py --fresh-process save` 建立原实现参考，再使用相同 `--output` 运行 `--fresh-process restore --score-offload`，对比下一次 GA4 Generator 更新及 optimizer state，保持两次 attention backend 相同。

训练异常先写 `failure-rank-*.json` 并打印 traceback，然后立即退出失败 worker，由 torchrun 结束其他 worker；异常路径不调用可能永久等待其他 rank 的 `destroy_process_group()`。正常完成仍正常清理。

### Fake 阶段卸载闲置模型

`runtime.offload_inactive_for_fake: true` 使用 `PhaseShardOffload`：Real 在连续5次Fake更新中保持CPU驻留；Generator 每个microbatch仅在no-grad生成x_hat时上GPU，之后立即卸载，直到下一个生成调用才恢复。现有recipe每次Fake训练都重新调用Generator，因此将Generator在整个5次Fake期间一直放CPU需要另行预计算样本；此实现不改变采样/RNG顺序、不复用x_hat。Generator阶段恢复Real/Generator，仍使用前述score反向卸载。

每个rank的 `active-microbatch-rank-XXXXX.json` 在前向前记录样本ID、阶段、GA索引及实际展开后的DiT token数。VLM图像占位符不重复计入文本token。更新日志增加 `actual_total_tokens` 和 `fake_phase_offload`。闲置优化器状态保持GPU，不属于此项卸载范围。

`runtime.early_dmd_checkpoint_steps: [5,40,45]` 可增加早期存盘，均在对应Generator更新完成后、debug之前保存完整DMD状态，原有周期存盘继续生效。
