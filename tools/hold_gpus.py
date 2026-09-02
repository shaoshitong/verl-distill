from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=70.0)
    parser.add_argument("--chunk-mib", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    bytes_per_chunk = args.chunk_mib * 1024 * 1024
    chunks_per_device = int(args.gib * 1024 / args.chunk_mib)
    tensors = []
    print(
        f"hold_gpus starting: devices={torch.cuda.device_count()} "
        f"target={chunks_per_device * args.chunk_mib} MiB/device",
        flush=True,
    )
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_device(device)
        device_tensors = []
        print(f"cuda:{device} allocating", flush=True)
        for _ in range(chunks_per_device):
            device_tensors.append(
                torch.empty(
                    (bytes_per_chunk // torch.tensor([], dtype=torch.float16).element_size(),),
                    dtype=torch.float16,
                    device=f"cuda:{device}",
                )
            )
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        tensors.append(device_tensors)
        print(
            f"cuda:{device} held={chunks_per_device * args.chunk_mib} MiB "
            f"free={free // 1024 // 1024} MiB total={total // 1024 // 1024} MiB",
            flush=True,
        )

    print("hold_gpus ready", flush=True)
    while tensors:
        time.sleep(60)
        print("hold_gpus heartbeat", flush=True)


if __name__ == "__main__":
    main()
