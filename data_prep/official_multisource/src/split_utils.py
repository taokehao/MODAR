from __future__ import annotations

import hashlib


def split_from_key(key: str, valid_pct: float = 0.01, test_pct: float = 0.01) -> str:
    """Deterministic split by record key, stable across reruns and data order changes."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + valid_pct:
        return "valid"
    return "train"
