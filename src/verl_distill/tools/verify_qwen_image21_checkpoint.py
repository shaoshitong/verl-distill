"""Read every DCP tensor chunk on CPU without materializing a full model or using GPUs.

Verifies committed inventory, tensor shapes/dtypes/finiteness, and cursor/RNG sidecars.
This is payload validation, not a substitute for distributed training-resume replay.
"""
import argparse
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from verl_distill.data.qwen_image21 import sha256, within, atomic_json


def verify(path, workers=4):
    path = Path(path)
    if path.name.endswith('.incomplete') or not (path/'COMPLETE').is_file():
        raise ValueError('Checkpoint is not committed')
    complete = json.loads((path/'COMPLETE').read_text())
    if complete.get('state_sha256') != sha256(path/'state.json'):
        raise ValueError('Checkpoint metadata digest mismatch')
    saved = json.loads((path/'state.json').read_text())
    for name, size in saved['files'].items():
        p = within(path,name)
        if not p.is_file() or p.stat().st_size != size:
            raise ValueError(f'Missing/truncated file: {p}')
    groups = defaultdict(list)
    for model in saved['models']:
        root = path/model
        metadata = FileSystemReader(root).read_metadata()
        for index, info in metadata.storage_data.items():
            tensor_meta = metadata.state_dict_metadata[index.fqn]
            groups[within(root,info.relative_path)].append((index,info,tensor_meta))
    def read_file(item):
        file, chunks = item
        tensors = byte_entries = tensor_bytes = 0
        with file.open('rb') as handle:
            for index, info, meta in sorted(chunks,key=lambda c:c[1].offset):
                handle.seek(info.offset)
                data = handle.read(info.length)
                if len(data)!=info.length:
                    raise ValueError(f'Truncated chunk: {file} {index.fqn}')
                if isinstance(meta,TensorStorageMetadata):
                    value = torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
                    expected = next(c for c in meta.chunks if c.offsets==index.offset)
                    if list(value.shape)!=list(expected.sizes) or value.dtype!=meta.properties.dtype:
                        raise ValueError(f'Invalid chunk shape/dtype: {file} {index.fqn}')
                    if value.is_floating_point() and not torch.isfinite(value).all():
                        raise ValueError(f'Nonfinite chunk: {file} {index.fqn}')
                    tensor_bytes += value.numel()*value.element_size()
                    tensors += 1
                    del value
                else:
                    # DCP stores non-tensor entries with torch.save as well.
                    torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
                    byte_entries += 1
        print(f'verified {file.name}: {tensors} tensor chunks, {tensor_bytes} tensor bytes',flush=True)
        return tensors,byte_entries,tensor_bytes
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read_file,groups.items()))
    for rank in range(saved['world_size']):
        extra=torch.load(path/f'rank-{rank:05d}.pt',map_location='cpu',weights_only=False)
        if not isinstance(extra.get('cursor'),dict) or not isinstance(extra.get('rng'),dict):
            raise ValueError(f'Invalid rank {rank} sidecar')
    return {'checkpoint':str(path),'updates':saved['updates'],'world_size':saved['world_size'],
            'files_verified':len(saved['files']),'tensor_chunks':sum(x[0] for x in results),
            'byte_entries':sum(x[1] for x in results),'tensor_bytes':sum(x[2] for x in results),
            'all_finite':True,'scope':'CPU payload readback; not distributed resume replay'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--report',required=True,help='Write outside the committed checkpoint directory')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if Path(args.report).resolve().is_relative_to(Path(args.checkpoint).resolve()):
        p.error('Report must be outside the immutable checkpoint')
    result=verify(args.checkpoint,args.workers)
    atomic_json(args.report,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
