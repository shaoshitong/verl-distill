#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
: "${QWEN21_TRAIN_MANIFEST:?Set the immutable train.json path}"
: "${QWEN21_EVAL_MANIFEST:?Set the matching eval.json path}"
: "${QWEN21_CONDITION_CACHE:?Set the published condition cache directory}"
: "${OUTPUT_DIR:?Set a fresh output directory}"
qwen_python="${QWEN21_PYTHON:-/tmp/qwen-image21-train-venv/bin/python}"
if [[ "${NNODES:-1}" == 1 ]]; then
    launch_args=(--standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-8}")
else
    : "${NODE_RANK:?Set NODE_RANK}"
    : "${RDZV_ENDPOINT:?Set host:port or [IPv6]:port}"
    : "${RDZV_ID:?Set a unique shared rendezvous id}"
    launch_args=(--nnodes="$NNODES" --nproc-per-node="${NPROC_PER_NODE:-8}"
                 --node-rank="$NODE_RANK" --rdzv-backend=c10d
                 --rdzv-endpoint="$RDZV_ENDPOINT" --rdzv-id="$RDZV_ID")
fi
exec "$qwen_python" -m torch.distributed.run "${launch_args[@]}" \
    -m verl_distill.cli.train --config \
    "$repo_root/configs/recipes/qwen_image21/dmd_hf_cfg2_fsdp1.yaml" "$@"
