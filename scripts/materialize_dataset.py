#!/usr/bin/env python3
"""Restore a released JSONL split in manifest order and verify its SHA-256."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    records = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
            records += block.count(b'\n')
    return digest.hexdigest(), records


def materialize(dataset_dir: Path, config: str, split: str, output_dir: Path) -> Path:
    entries = json.loads((dataset_dir / 'SPLITS.json').read_text())
    matching = [e for e in entries if e['configuration'] == config and e['split'] == split]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one manifest entry for {config}/{split}')
    entry = matching[0]
    expected = (entry['source_sha256'], entry['records'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'{split}.jsonl'
    if output.exists():
        if fingerprint(output) != expected:
            raise ValueError(f'Existing file differs from the release; not overwritten: {output}')
        return output

    digest = hashlib.sha256()
    records = 0
    root = dataset_dir.resolve()
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f'.{split}.', delete=False) as target:
        temporary = Path(target.name)
        try:
            for shard in entry['shards']:
                path = (root / shard['file']).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Shard path is outside the downloaded dataset')
                with gzip.open(path, 'rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                        records += block.count(b'\n')
                        target.write(block)
            if (digest.hexdigest(), records) != expected:
                raise ValueError(f'Checksum or record count mismatch for {config}/{split}')
            target.flush()
            # Publish without overwriting a file created by a concurrent process.
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path('artifacts/data'))
    parser.add_argument('--config', choices=['tier-a', 'mp20'], required=True)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = materialize(args.dataset_dir, args.config, args.split, args.output_dir)
    print(f'Verified JSONL: {output}')


if __name__ == '__main__':
    main()
