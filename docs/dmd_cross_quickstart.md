# 新节点复现：Schedule-Free generator 与浅层 DMD-STE

本页覆盖实际交叉消融的 A/B 两组；[算法与 FSDP1 细节](dmd_cross_reproduction_zh.md)以 A 的六层 gen 目标为主。

| 组 | fake 目标 | generator STE 权重 |
|---|---|---|
| A | 普通 latent x0 MSE | latent、block 1–5 全为 1 |
| B | latent + block 1–5 特征 MSE，单位权重 | latent=1，其余=0 |

共同设置：generator Schedule-Free AdamW，lr=5e-5、betas=(0,0.999)、clip=2.83；fake AdamW，lr=1e-5、clip=1；live anchor，4 fake + 1 gen，8 GPU/FSDP1，每卡 microbatch=1、累积4；从同一 step-1000 model-only checkpoint 独立训练到2500。gen MSE surrogate 保留本实验系数1，并非普通 DMD 分支的0.5。B 中五层原始 gen loss 仍记录，但权重0，不参与总损失。

## 1. 拉取与安装

```bash
git clone https://github.com/shaoshitong/verl-distill.git
cd verl-distill
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]' -c requirements/dmd_cross_constraints.txt
```

约束文件记录关键历史版本，包括 transformers 4.57.0；它用于复现实验，不是最新版本推荐。CUDA 驱动必须支持所安装的 PyTorch wheel。Schedule-Free AdamW 实现在本仓库内，不额外依赖 pip 的 schedulefree 包。

## 2. 准备资源并设置路径

Git 不包含模型、训练数据或 step-1000 checkpoint。新节点需要自行挂载/下载这些资源；只有代码无法重建原起点。数据发布说明见主 README。

```bash
export ZIMAGE_MODEL_PATH=/path/to/Z-Image-FM1
export ZIMAGE_FAKE_SCORE_MODEL_PATH=/path/to/Z-Image-FM1
export ZIMAGE_TEACHER_MODEL_PATH=/path/to/Z-Image
export TRAIN_LANCE_DATA_DIR=/path/to/zimage_merged_notext_turbogen_lance
export ZIMAGE_ODE_PAIR_DIR=/path/to/ode_pairs
export RESUME_FROM=/path/to/checkpoints/step-1000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

即使从1000恢复，当前 trainer 仍初始化 ODE pair loader，其资源也必须提供。恢复不加载 optimizer state，也不精确恢复 dataloader 游标。`RESUME_FROM` 是同一份完整的分片模型起点，目录应包含 `.metadata` 和各 rank 的 `.distcp` 文件。

## 3. 检查配置并准备新队列

```bash
python -m verl_distill.cli.train --config dmd_cross_fake0_gen0to5 --dry-run
python -m verl_distill.cli.train --config dmd_cross_fake0to5_gen0 --dry-run

# 可选：全部成功后占卡。脚本由新节点提供；省略则队列结束后退出。
export GPU_OCCUPIER_SCRIPT=/path/to/auto_occupy_gpu.py
python scripts/prepare_dmd_cross_ablation.py --output-root /path/to/runs/cross_ablation_new
```

准备命令只生成配置/脚本，不开始训练。生成目录必须不存在，不能把原实验输出目录用作重跑目录。它会冻结环境变量解析后的 YAML，并记录源代码 hash。

确认8张GPU可用后，通过持久终端或后台启动：

```bash
nohup bash /path/to/runs/cross_ablation_new/run_sequential.sh > /path/to/runs/cross_ablation_new/launcher.log 2>&1 < /dev/null &
```

A 完成并通过校验后才启动 B。队列检查完整步数、更新顺序、梯度有限性、gen 分层加权和、最终 checkpoint。错误会记在 `queue_state.json`；不会自动跳过错误进入下一组。两组间不运行占卡。

## 4. 输出与日志解释

```text
queue_state.json
queue.log
occupy.log                      # 配置了占卡且两组成功时才生成
A_fake0_gen0to5/
  resolved_config.yaml
  train.log
  dmd_stats.tsv
  debug_samples/step-XXXXXX/trajectory.png
  checkpoints/step-2500/
B_fake0to5_gen0/
  ...
```

每25步采样，每500步保存。每组应包含1200次fake与300次gen更新。

gen TSV 的六列为 `gen/teacher_feature_loss_latent` 和 `gen/teacher_feature_loss_layer_1` 到 `layer_5`；同时记录分母和RMS。A总损失等于六项之和，B总损失只等于latent项。未更新角色的零值不能混入对应角色的均值。

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
bash scripts/check_public_tree.sh
```

运行中的源代码不要随意修改，队列在下一组启动前会核对 hash。再次运行请选择新目录，并记录实际 Git commit。
