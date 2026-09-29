# Qwen-Image-2.1 DMD

## 当前配方

[`dmd_hf_cfg2_fsdp1.yaml`](../configs/recipes/qwen_image21/dmd_hf_cfg2_fsdp1.yaml)
从 HF 权重直接启动 DMD，`reflow_updates=0`。Generator、Fake、Teacher 为独立参数副本，
Teacher 始终冻结；新实验使用新的优化器、随机数状态和数据游标。

| 配置 | 值 |
| --- | --- |
| 训练量 | 3000 次 Fake 更新，5 Fake → 1 Generator，共 600 次 Generator 更新 |
| Generator AdamW | LR=5e-7，betas=(0, 0.999)，weight_decay=0.1 |
| Fake AdamW | LR=2e-6，betas=(0.9, 0.95)，weight_decay=0.1 |
| Loss / grad clip | Generator、Fake 的 loss weight 和 grad clip 均为 1 |
| Guidance | Teacher CFG=2；Generator、Fake 仅正向条件 |
| 并行 | 4节点×8卡，每卡 microbatch=1，GA=1，每次更新全局 batch=32 |
| 精度 | BF16 forward，FP32 主参数/梯度/reduction，FP64 DMD surrogate |
| 参数 | 仅 transformer_blocks 可训练；time/text embedding 等保持冻结 |
| 显存 | 三个模型各32个block独立包裹FSDP1；use_orig_params=true；训练模型开启非重入gradient checkpointing；两项offload关闭 |

无 LR scheduler、warmup、EMA 或 GAN。每个完整周期消费 192 条样本。
旧 REFLOW 配方仍可显式选择，其优化器和 GA 配置以对应 YAML 为准。

## 环境

使用独立的 Python 3.12 环境：

```bash
python3.12 -m venv /path/to/qwen-venv
source /path/to/qwen-venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[qwen21,dev]'
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

`qwen21` extra 固定 Transformers 版本并限制 Diffusers 版本范围，包含 debug 绘图依赖。
模型使用 `Qwen/Qwen-Image-2.1`；如需严格固定公开模型或 Diffusers revision，
通过 `QWEN21_MODEL_REVISION` 和 `QWEN21_DIFFUSERS_REVISION` 设置。模型目录需包含
transformer、text_encoder、processor、vae、scheduler。
每个节点分别准备环境和本地模型/参考图副本，并使用相同代码与数据身份。

## 数据和条件缓存

生产数据由 `complete.json`、`initial_noise.pt`、`x0_latent.pt`、`image.png` 组成。
快照校验文件哈希，保留样本尺寸和参考图顺序；训练 latent 已归一化。
评测 CSV 需有64条唯一记录，列为 `prompt_id,prompt,width,height`；debug 使用前8条。

```bash
export QWEN21_MODEL_PATH=/path/to/Qwen-Image-2.1
export QWEN21_REFERENCE_ROOT=/path/to/extracted_references
export QWEN21_DEBUG_CSV=/path/to/complex_prompt.csv
export QWEN21_TRAIN_MANIFEST=/path/to/snapshot/train.json
export QWEN21_EVAL_MANIFEST=/path/to/snapshot/eval.json
export QWEN21_CONDITION_CACHE=/path/to/conditions/positive
export QWEN21_NEGATIVE_CONDITION_CACHE=/path/to/conditions/negative

python -m verl_distill.tools.prepare_qwen_image21 snapshot \
  --output-root /path/to/production_outputs \
  --reference-root "$QWEN21_REFERENCE_ROOT" \
  --prompts-csv "$QWEN21_DEBUG_CSV" --snapshot-dir /path/to/snapshot

python -m verl_distill.tools.prepare_qwen_image21 cache \
  --model "$QWEN21_MODEL_PATH" --manifest "$QWEN21_TRAIN_MANIFEST" \
  --eval-manifest "$QWEN21_EVAL_MANIFEST" --cache-dir "$QWEN21_CONDITION_CACHE" \
  --reference-root "$QWEN21_REFERENCE_ROOT" --num-shards 1 --shard-id 0
python -m verl_distill.tools.prepare_qwen_image21 publish \
  --manifest "$QWEN21_TRAIN_MANIFEST" --eval-manifest "$QWEN21_EVAL_MANIFEST" \
  --cache-dir "$QWEN21_CONDITION_CACHE" --num-shards 1

for command in build publish; do
  CUDA_VISIBLE_DEVICES=0 python -m verl_distill.models.qwen_image21.negative_cache "$command" \
    --model "$QWEN21_MODEL_PATH" --manifest "$QWEN21_TRAIN_MANIFEST" \
    --eval-manifest "$QWEN21_EVAL_MANIFEST" --output "$QWEN21_NEGATIVE_CONDITION_CACHE" \
    --reference-root "$QWEN21_REFERENCE_ROOT" --shards 1 --shard 0
done
```

已有匹配的快照和缓存可直接复用。多GPU构建时，各进程使用相同分片总数和不同分片编号，
全部结束后再 publish。负向条件为空文本，保留相同参考图及顺序；相同参考图组合会复用编码，
因此负向条件数量可以小于样本数。CFG强度不参与编码，CFG=2和CFG=4可共用条件缓存。

快照、原始payload、缓存及 `reused_cache` 引用链都必须保存在持久数据目录，保持不可变，
不要放进需要清理的实验输出目录。VLM只用于离线编码；VAE在debug时加载。

## 启动

设置上述数据变量后，在每个节点执行一次：

```bash
export OUTPUT_DIR=/path/to/runs/qwen_hf_cfg2
export QWEN21_PYTHON=/path/to/qwen-venv/bin/python
export QWEN21_RESUME_FROM=""
export NNODES=4 NPROC_PER_NODE=8
export NODE_RANK=0  # 四个节点分别为0、1、2、3
export RDZV_ENDPOINT=HOST:PORT  # IPv6使用[ADDRESS]:PORT
export RDZV_ID=qwen_hf_cfg2

"$QWEN21_PYTHON" -m verl_distill.cli.train --dry-run \
  --config configs/recipes/qwen_image21/dmd_hf_cfg2_fsdp1.yaml
bash scripts/train_qwen_image21.sh
```

所有节点使用相同 rendezvous、共享输出目录、配置和缓存。按集群实际网络设置
`NCCL_SOCKET_IFNAME`、`NCCL_SOCKET_FAMILY`、`GLOO_SOCKET_IFNAME`；确认NCCL走预期的IB路径。
单节点可设置 `NNODES=1`。其他配方通过 `--config` 显式指定。
每次启动都使用新输出目录，运行配置保存在 `run.json`。

## 时间步和损失

训练时均匀选择六个Generator sigma之一，给数据x0重新加噪，做一次Generator前向：

```text
x_t = (1-t) * clean + t * noise_g
y = x_t - t * G(x_t, t, positive)
```

六点继承Qwen按尺寸计算的动态shift，最后一次模型调用在sigma=0.4，积分终点为0。
模型直接接收sigma，不做 `1-sigma` 反转。完整六次Euler调用用于评测。
Score sigma独立采样自原始1000点网格中[0.02,0.98]的点。

```text
z = (1-s) * stopgrad(y) + s * noise
Fake loss = mean((F(z,s,positive) - (noise-stopgrad(y)))²)
v_real = v_negative + 2 * (v_positive-v_negative)
fake_x0 = z - s * v_fake
real_x0 = z - s * v_real
direction = (fake_x0-real_x0) / max(mean(abs(stopgrad(y)-real_x0)), 1e-6)
Generator loss = 0.5 * mean((y-stopgrad(y-direction))²)
```

上述mean按样本计算。Fake使用恒定权重1的velocity MSE，高噪声点不降权；
`fake_min_alpha`、`fake_max_weight` 只影响旧的 `epsilon_via_x0_capped` 模式。
Generator更新时Fake/Teacher前向无梯度，DMD重建和差值采用FP64。

## 输出与恢复

- `logs/rank-*.jsonl`：每次更新的loss、梯度、LR、sigma、token数、耗时和显存。
- `checkpoints/fake_step_*`：完整模型、AdamW、计数、RNG和数据游标；当前配方在5/40/45、每600次Fake更新及结束时保存。
- `debug/fake_step_*/rollout/`：每20次Fake更新后，固定8条prompt的6NFE输出。
- `debug/fake_step_*/rollout_25step/`：相同prompt/噪声的25步对照，学生仅正向条件。
- `debug/fake_step_*/train_dmd/`：实际训练样本的Generator/Fake/Real x0及signed/absolute diff；Real使用Teacher CFG。
- `dmd-initialization-rank-*.json`、`infra-rank-*.json`、`score-infra-rank-*.json`：初始化来源、FSDP包裹、可训练参数和CFG。

保存发生在对应Generator更新之后、debug之前。只有带 `COMPLETE` 标记的checkpoint可恢复；
设置 `QWEN21_RESUME_FROM`，保持world size和数值配置一致，并使用新的 `OUTPUT_DIR`。
`init_reflow_from` 是单独的REFLOW权重初始化路径，不恢复优化器，与 `resume_from` 互斥。

显式缓存重建可设 `allow_condition_cache_rebuild: true`，只允许条件索引变化，要求数据清单、
模型、优化器等身份一致；变更offload等基础设施配置需另设 `allow_infra_resume_change: true`。
两种迁移均记录在 `infra_migration.json`。重新编码不保证与已删除的缓存逐字节相同。

导出Generator时使用checkpoint原world size运行 `verl_distill.tools.export_qwen_image21`，
传入 `--checkpoint`、`--model` 和新的 `--output`。输出为Diffusers transformer目录，
其余组件继续使用原始HF模型。
