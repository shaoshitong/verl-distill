# Generator Schedule-Free AdamW + 0–5 层 DMD-STE

本文面向接手实现、复现实验的 agent，描述实际执行版本，不把建议修改混入历史实验。


## 1. 实验配置

| 项目 | A_fake0_gen0to5 |
|---|---|
| fake score 目标 | 普通 latent x0 去噪 MSE |
| generator 目标 | latent + teacher block 1–5 的 normalized STE |
| generator 权重 | 六项全为 1 |
| fake 特征权重 | 不参与普通 MSE 分支 |
| generator anchor | live，梯度在当前生成 latent 处通过 teacher |
| generator optimizer | Schedule-Free AdamW |


## 2. “相较于原始 DMD”的准确含义

理论参考：[One-step Diffusion with Distribution Matching Distillation, §3.2](https://arxiv.org/html/2311.18828v3#S3.SS2)。原文式 1 的目标是 KL(p_fake || p_real)，式 7 用 real/fake score 差构成 generator 梯度；fake 去噪器使用式 6 的生成样本去噪回归。**不是对两个去噪器预测之间的普通 MSE 继续反传。**

工程对照是本项目的普通 latent DMD 分支 `StandardDMD._generator_dmd_loss_impl`，以及历史运行 `dmd_refaligned_fsdp1_ode_warmup_50000_20260902_v2`。该运行 `train.log` 记录 generator 学习率 `1e-5`，本次为 `5e-5`，即 **5 倍**。不要把 `1e-5` 说成论文所有设置统一采用的学习率。

主要改动：

1. generator 用本地实现的 Schedule-Free AdamW，学习率从上述工程对照的 1e-5 提到 5e-5；fake 用普通 AdamW 1e-5。
2. A 组的 gen 把 latent 方向扩展到冻结 teacher 的 1–5 层特征方向，逐层归一化后注入梯度。
3. 采用明确的 4 fake + 1 gen 交替周期，每次都是完整的梯度累积窗口。

## 3. 核心一：generator 的 Schedule-Free AdamW

### 3.1 实际超参数

```yaml
optimizer:
  generator:
    type: adamw_schedule_free
    lr: 5.0e-5
    betas: [0.0, 0.999]
    weight_decay: 0.01
    max_grad_norm: 2.83
    lr_scheduler: none
    warmup_lr: 1.0e-4
    warmup_betas: [0.9, 0.999]
  fake_score:
    type: adamw
    lr: 1.0e-5
    betas: [0.9, 0.999]
    weight_decay: 0.0
    max_grad_norm: 1.0
    lr_scheduler: none
```

Schedule-free 构造函数的未显式指定值：`eps=1e-8, warmup_steps=0, r=0, weight_lr_power=2, inner_momentum=0, foreach=True`。

注意区分三件事：

- `warmup_lr/warmup_betas` 是 ODE 预训练阶段配置；本次从 step-1000 开始，不重跑该阶段。
- optimizer 的 `warmup_steps` 为 0；不是再做 1000 步学习率 warmup。
- `lr_scheduler: none` 表示没有外部 cosine 等 scheduler。配置残留的 `lr_decay_steps/lr_min` 在这里不生效。

`betas[0]=0` 是这次实验真实设置，不能随手换成常见的 0.9。这里的 beta1 控制 schedule-free 的训练/平均权重关系，不能只按普通 Adam 的一阶动量解释。实现另有 `inner_momentum`，本次为 0。

### 3.2 实现在哪里，状态是什么

使用仓库自己的 `src/verl_distill/optim/schedulefree.py:AdamWScheduleFree`，不是任意 pip 版本的同名 optimizer。

每个参数保存：

- `z`：更新轨迹参数；
- `x`：beta1=0 时显式保存的平均参数；
- `exp_avg_sq`：二阶矩；
- `exp_avg`：仅 inner_momentum 非零时存在，本实验没有。

参数组还保存 `k, weight_sum, lr_max, train_mode, scheduled_lr` 等。`k` 是该 optimizer 实际 step 的计数，不是训练的 global_step；这段 1500 个全局 step 只有 300 次 generator optimizer.step。

在本实验 beta1=0、固定学习率、r=0 的情况下，可用以下等价简化理解（实际实现有 foreach/scalar 两种分支）：

```text
n = k + 1
v = beta2 * v + (1-beta2) * g^2
g_adapt = g / (sqrt(v / (1-beta2^n)) + eps)
z_new = z - lr * (g_adapt + weight_decay * z)
c = 1/n
x_new = (1-c) * x + c * z_new
training_parameter = z_new
k = k + 1
```

更一般的平均权重系数是：

```text
weight_n = n^r * lr_max^weight_lr_power
c_n = weight_n / accumulated_weight_sum
```

本次每个 generator 更新学习率相同，因此平均相当于对更新后的 z 序列做等权平均；**不是另设固定衰减率的 EMA**。

`optimizer.train()` 在 beta1=0 时把 `z` 拷到模型参数；`optimizer.eval()` 把 `x` 拷到模型参数。`model.train()/eval()` 只改变模块模式，不能代替这两个优化器调用。

由此产生一个实际行为：训练期间用于生成训练样本的是训练权重 z；固定提示词采样和导出 checkpoint 的模型权重是平均权重 x。

### 3.3 如何与 FSDP1 配合

关键代码：

- `engine/fsdp1.py:apply_zimage_fsdp1`
- `trainers/dmd.py:_build_optimizer, _optimizers_eval_context, _clip_grad_norm_for, train_dmd`
- `engine/checkpoint.py:save_distributed_training_state, load_distributed_training_state`

**顺序不能颠倒：**

1. 加载 generator、real score、fake score 的完整模型。冻结 real score 参数；generator/fake 可训练。
2. 在 `apply_zimage_fsdp1` 内先 `module.float()`，然后包装 FSDP1。
3. 在包装后的模块上收集 `requires_grad=True` 的参数，再构造 optimizer。
4. 每个 rank 的 schedule-free 状态直接建立在该 rank 的本地参数分片上。

FSDP1 的关键选项：

```python
FSDP(
    module,
    auto_wrap_policy=block_class_based_policy,
    device_id=local_rank,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    use_orig_params=False,
)
```

这里 `use_orig_params=False` 使 optimizer 处理 FSDP 的扁平参数分片；FP32 主参数/状态与 BF16 前向计算相结合。`z/x/exp_avg_sq` 通过本地 parameter 的 clone/zeros_like 初始化，形状和 device 跟随分片。不要另外构造一份全模型 optimizer，也不要每步 all-gather 全部 optimizer 状态。

梯度由 FSDP 同步和分片；optimizer 是逐元素的本地分片更新。每次更新前，使用 FSDP 模块的 `clip_grad_norm_` 求全局范数并裁剪，而不是只裁本 rank 的局部范数。该仓库主动拒绝把此 schedule-free 分支用于 FSDP2；不要在复现时只把 backend 字符串改成 fsdp2。

训练生命周期的伪代码：

```python
# 所有 rank 执行
wrapped_generator = apply_zimage_fsdp1(generator, ...)
opt = AdamWScheduleFree(wrapped_generator.parameters(), lr=5e-5, betas=(0, .999), ...)
opt.train()

# 仅 generator 轮次执行 opt.step；四个 fake 轮次不更新它
opt.zero_grad()
for microbatch in four_microbatches:
    # 前三个窗口使用 FSDP no_sync，最后一个同步
    with appropriate_gradient_sync_context():
        loss = generator_loss(...)
        (loss / 4).backward()
wrapped_generator.clip_grad_norm_(2.83)
opt.step()

# 采样与保存前，每个 rank 都必须切换平均参数
opt.eval()
try:
    sample_or_save_distributed_checkpoint(...)
finally:
    opt.train()
```

不要只在 rank0 上调用 optimizer.eval：FSDP 需要各 rank 共同组成同一套平均权重。采样/保存期间也不要执行 optimizer.step。

## 4. 核心二：gen 的 0–5 层 normalized DMD-STE

### 4.1 各变量与层编号

- `u`：本轮 generator 生成的干净 latent；保留到 generator 参数的计算图。
- `sigma`：DMD score query 的噪声量；与 feature timestep 不是同一概念。
- `q = (1-sigma) * u + sigma * noise`：real/fake 两个 score 共用的加噪 query。
- `r`：real score 对 q 预测的干净 latent；CFG=5.5。
- `f`：fake score 对同一个 q 预测的干净 latent；CFG=0。
- `H_k(v)`：把 v 输入冻结 real teacher，在固定 feature timestep=0.2 和同一文本条件下提取第 k 个主 transformer block 的图像 token 特征。
- `H_0(v)=v`：latent 恒等映射，**不是 transformer.layers[0]**。

1th 对应主干 `self.layers[0]` 的输出，5th 对应 `self.layers[4]` 的输出。它们不指 noise/text refiner 的层，也不指五个扩散时间步。

当前实际生成输入路径使用 `StandardDMD.generate_one_step_latents`：它沿用项目的四步采样/训练时间点选择逻辑。复现时应复用该函数、`_sample_sigmas` 和 flow→x0 转换函数 `_predict_x0_from_flow`，不要把 `num_denoising_step=4` 简化成任意新的一步时间采样方案。

### 4.2 如何取出 1–5 层

调用链：

```text
FullModelDMD（继承 generator_loss）
  → StandardDMD._generator_teacher_feature_loss_impl
    → _call_teacher_representations
      → _call_teacher_features
        → GenTransformer.forward(feature_layers=(1,2,3,4,5))
          → ZImageTransformer2DModelWrapper.forward
```

在 `models/zimage/transformer.py` 主 block 循环中：

```python
layer_number = layer_idx + 1
if layer_number in selected_layers:
    features[slot_name] = unified[:, :image_token_count]
```

取的是完整图像 token 特征 `[B, N_image, C_hidden]`，不是全局池化向量。文本参与 joint attention，但输出损失只用图像 token 切片。底层按所选层的槽位命名，`_call_teacher_features` 将槽位名映射回实际层编号。

`_call_teacher_representations` 再补上 `{'latent': input_latent}`，latent 形状 `[B,C,H,W]`。`teacher_feature_include_last=false` 排除 pre_projector 分项；底层虽返回这个字段，上层不选入损失。当前实现没有在 block 5 后提前退出 teacher 主干，不要把它误述成截断到五层的 teacher 网络。

### 4.3 哪些路径保留梯度

当前 `generator_teacher_feature_anchor=live`，流程为：

1. 保留 u=G(...) 的计算图。
2. `_gan_score_pair` 在 no_grad 下计算相同 q 的 r、f，二者 detach。
3. no_grad 下提取 `H_k(r)`、`H_k(f)`。
4. 开启 autograd 提取 `H_k(u)`。teacher 参数保持 `requires_grad=False`，但**不能对这次 teacher 前向使用 no_grad**，因为需要对 teacher 输入的梯度。
5. 每层在自己的表征空间构造 detached 方向与 surrogate target。

A/B 的 generator 都不会穿过 real/fake 去噪预测 r/f 的网络 Jacobian。特征层会穿过 live 分支 `H_k(u)` 的输入 Jacobian。real anchor 则会换这个 Jacobian 的求值点，本次不是该设置。

### 4.4 实际公式（包括系数、精度和 reduction）

对于每一层 k，每个样本 b：

```text
h = H_k(u)
a = stopgrad(H_k(r))
b_fake = stopgrad(H_k(f))
denom = clamp_min(mean_nonbatch(abs(stopgrad(h)-a)), 1e-6)
direction = (a-b_fake) / denom
target = stopgrad(h) + direction
L_k = mean_all_elements((h-target)^2)
```

实现将 h/a/b_fake 转为 float64 后计算这些差值、分母和平方；前向 teacher 特征本身仍来自 BF16 计算。这不是对 teacher 特征先做 L2 单位化，也不是除以各层特征 RMS。**归一化的是 detached 方向，分母来自 live 与 real 的平均绝对差。**

A：`L_gen = L_0 + L_1 + L_2 + L_3 + L_4 + L_5`。是六项直接求和，不除以 6。

B：`L_gen = L_0`，其他五项日志保留但乘 0。`teacher_feature_grad_balance=null`，没有动态梯度平衡。

单样本方向的符号是 real minus fake；对 h 的梯度为 `-2*direction/numel(h)`。因此梯度下降使 h 朝 real-fake 方向走。不要把这个符号与“梯度张量”写反。

0th 的 Jacobian 为 I，直接产生 latent 方向；k>0 的梯度还要乘 `J_Hk(u)^T`。这就是特征扩展改变更新方向的地方，并非只对原始 latent loss 加了几个标量系数。

**保留历史系数：**普通 DMD 分支 `_generator_dmd_loss_impl` 用 `0.5*MSE`；本特征 STE 分支（包括 latent）用 `MSE`。在相同 r/f/分母的有限值条件下，0th 的方向一致，loss 与梯度幅度为前者两倍（另有计算精度等实现差异）。复现本实验必须保持这个系数；若改成 0.5，应作为另一个实验明确记录，不能静默“修正”。

`dm_loss_weight=1` 在本配置中不造成额外缩放；该特征分支实际通过各层权重求和，复现时应以 `_generator_teacher_feature_loss_impl` 为准，不要假定普通 DMD 分支的所有附加 loss 开关都会自动生效。

### 4.5 可直接核对的最小损失伪代码

```python
# u retains generator graph. teacher weights frozen.
with torch.no_grad():
    pair = same_noisy_query_real_fake_predictions(u.detach(), ...)
    real_features = representations(pair['r'])
    fake_features = representations(pair['f'])
live_features = representations(u)  # NOT in no_grad

losses = {}
for key in ('latent', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5'):
    h = live_features[key].double()
    real = real_features[key].detach().double()
    fake = fake_features[key].detach().double()
    dims = tuple(range(1, h.ndim))
    denom = (h.detach() - real).abs().mean(dims, keepdim=True).clamp_min(1e-6)
    direction = (real - fake) / denom
    target = h.detach() + direction
    losses[key] = (h - target).square().mean()
loss = sum(weights[key] * value for key, value in losses.items())
```

这段只是损失构造，不代替 Z-Image 的模型封装、flow parameterization、conditioning、CFG 与 FSDP 调用。

## 5. Fake score 的两种目标

实际入口是 `algorithms/dmd/full_model.py:FullModelDMD.score_loss`，不要只查看 StandardDMD 的 score_loss 而遗漏全模型覆盖。

先在 no_grad 下生成 u、加噪，再由可训练 fake score 输出 p=pred_x0。


```text
score_objective = mse
score_loss_target = x0
score_use_weighting = false
L_fake = mean((p-u.detach())^2)
```
也就是标准的fake model更新


## 6. 共同训练设置与启动

| 项目 | 设置 |
|---|---|
| GPU | 单机 8×A100 80GB；FSDP1 / NCCL |
| 模型 | generator/fake 初始化目录 Z-Image-FM1；real teacher Z-Image |
| 模型恢复 | generator 和 fake 从同一 step-1000 分片 checkpoint 恢复 |
| 数据 | image_lance，9529 条；refined_prompt；center_crop=true，random_flip=false |
| 分辨率桶 | 1024×1024、832×1408、1408×832 |
| 每轮 microbatch | 每卡 1，梯度累积 4；8 卡合计 32 个样本/更新窗口 |
| 训练范围 | global_step=1001…2500 |
| 更新顺序 | 1001–1004 fake，1005 gen，之后每 5 步重复 |
| 更新次数 | 每组 1200 次 fake、300 次 gen |
| precision | BF16 前向，FP32 梯度规约/主参数；特征损失内部 FP64 |
| sampling | 4 步，debug CFG=0，seed=42，每 25 步生成固定提示词 |
| 保存 | step-1500/2000/2500 |
| discriminator / GAN | 未启用；配置中残留 GAN 默认参数不代表启用 |

`dfake_gen_update_ratio=4` 在 teacher-feature trainer 路径中是“四次 fake 后一次 gen”。不要仅照抄 `method.should_update_generator(step)` 的取模逻辑；trainer 对该路径有覆盖：

```python
phase = (global_step - warmup_iterations - 1) % (fake_updates + 1)
update_score = phase < fake_updates
update_generator = phase == fake_updates
```

模型、数据、ODE pair 和 checkpoint 绝对路径均在两份 `resolved_config.yaml` 内。即使从 step-1000 恢复且不再做 ODE warmup，当前 trainer 仍构造 ODE pair loader，所以其目录也必须可用。


## 7. 源码导航

路径均相对于代码根目录：

| 文件/函数 | 作用 |
|---|---|
| `optim/schedulefree.py`（位于 src/verl_distill） | 本地 Schedule-Free AdamW，z/x、train/eval、lazy state |
| `engine/fsdp1.py:apply_zimage_fsdp1` | FSDP1 包装、混合精度、FlatParameter |
| `engine/checkpoint.py` | 分片模型/optimizer 保存恢复及 lazy state 初始化 |
| `trainers/dmd.py:_build_optimizer` | optimizer 构造与 FSDP1 限制 |
| `trainers/dmd.py:_optimizers_eval_context` | 全 rank 切换平均权重 |
| `trainers/dmd.py:train_dmd` | 包装顺序、optimizer 构造、更新轮次、GA、采样存盘 |
| `trainers/dmd.py:_tracked_stats_keys` | 动态预留分层 TSV 列 |
| `algorithms/dmd/method.py:_teacher_pair` | 同一 noisy query 的 real/fake 去噪预测 |
| `algorithms/dmd/method.py:_call_teacher_representations` | latent 与选中 teacher block 的组合 |
| `algorithms/dmd/method.py:_teacher_feature_ste_losses` | 分母、detached 方向、MSE surrogate |
| `algorithms/dmd/method.py:_generator_teacher_feature_loss_impl` | gen 多层前向/梯度组织 |
| `algorithms/dmd/full_model.py:FullModelDMD.score_loss` | A/B fake 目标实际分支 |
| `models/zimage/modeling.py:GenTransformer.forward` | 传递 feature_layers，保持 Z-Image 输入封装 |
| `models/zimage/transformer.py:forward` | 主干 block 编号与图像 token 特征截取 |

新节点安装、A/B 配置及串行启动方法见 [快速开始](dmd_cross_quickstart.md)。
