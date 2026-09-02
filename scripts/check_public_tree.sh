#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

patterns='code\.byted|bytedpypi|/mnt/(hdfs|shared_data)|__MERLIN_USER_DIR__|hdfs://|s3://|\bs[0-9]{2,4}\b|\b[0-9a-f]{40}\b|api[_-]?key[[:space:]]*[:=]|password[[:space:]]*[:=]|secret[[:space:]]*[:=]'

if rg -n -i "$patterns" \
  --glob '!scripts/check_public_tree.sh' \
  --glob '!.git/**' \
  --glob '!*.lock' .; then
  echo "Public-tree scan found a blocked private-infrastructure pattern." >&2
  exit 1
fi

blocked_files="$(find . -type f \( \
  -name '*.ckpt' -o -name '*.safetensors' -o -name '*.distcp' -o \
  -name '*.pt' -o -name '*.pth' \) -print)"
if [[ -n "$blocked_files" ]]; then
  echo "$blocked_files" >&2
  echo "Public-tree scan found a blocked model or checkpoint artifact." >&2
  exit 1
fi

echo "Public-tree scan passed."
