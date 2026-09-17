from __future__ import annotations

import gzip
import os
import pickle
from collections.abc import Iterator
from typing import Any

import lmdb


def dumps_record(record: dict[str, Any], compress: bool = True) -> bytes:
    payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    return gzip.compress(payload) if compress else payload


def loads_record(payload: bytes, compressed: bool = True) -> dict[str, Any]:
    if compressed:
        payload = gzip.decompress(payload)
    return pickle.loads(payload)


def iter_lmdb(path: str, compressed: bool = True) -> Iterator[tuple[bytes, dict[str, Any]]]:
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                yield key, loads_record(value, compressed=compressed)
    finally:
        env.close()


class LmdbWriter:
    def __init__(
        self,
        path: str,
        map_size: int = 256 * 1024**3,
        commit_every: int = 1000,
        compress: bool = True,
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.commit_every = commit_every
        self.compress = compress
        self.env = lmdb.open(path, subdir=False, map_size=map_size, lock=False)
        self.txn = self.env.begin(write=True)
        self.count = 0

    def put(self, record: dict[str, Any], key: bytes | None = None) -> None:
        if key is None:
            key = str(self.count).encode("ascii")
        self.txn.put(key, dumps_record(record, compress=self.compress))
        self.count += 1
        if self.count % self.commit_every == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        self.env.sync()
        self.env.close()

    def __enter__(self) -> "LmdbWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
