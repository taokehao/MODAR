#!/usr/bin/env python
"""Download the opXRD subsets that actually have atomic coordinates.

The Zenodo dump is 1.4 GB and ~92k JSON patterns. Hollarek et al. Table 1
says atomic coordinates exist only for:

* CNRS  — 1052 patterns, ~85% with coordinates (COD-extracted)
* HKUST-A — 21 patterns with CIF labels

INT / IKFT / LBNL are unlabeled. EMPA / USC have composition or lattice
but not atom sites, so they cannot be scored as CSP ground truth.

This script range-fetches those two zip regions (≈49 MB + 0.4 MB) and
extracts JSON, avoiding the full dump.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OPXRD_URL = "https://zenodo.org/api/records/15298026/files/opxrd.zip/content"
ZIP_SIZE = 1_400_997_733
CD_OFF = 1_390_725_753
CD_SIZE = 10_271_882
UA = "multimodal_csp-v2/opxrd-labeled"
KEEP_PREFIXES = ("CNRS/", "HKUST/HKUST-A/")
CHUNK = 4 * 1024 * 1024
WORKERS = 6
RETRIES = 8


def http_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": UA,
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            expect = end - start + 1
            if len(data) != expect:
                raise IOError(f"short read {len(data)} != {expect} for {start}-{end}")
            return data
        except Exception as exc:
            last = exc
            time.sleep(min(30, 1.5 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def parse_central_directory(blob: bytes) -> list[dict]:
    off = 0
    rows: list[dict] = []
    while off + 46 <= len(blob):
        if blob[off : off + 4] != b"PK\x01\x02":
            nxt = blob.find(b"PK\x01\x02", off + 1)
            if nxt < 0:
                break
            off = nxt
            continue
        method = struct.unpack_from("<H", blob, off + 10)[0]
        csize, usize = struct.unpack_from("<II", blob, off + 20)
        nlen, elen, clen = struct.unpack_from("<HHH", blob, off + 28)
        loff = struct.unpack_from("<I", blob, off + 42)[0]
        name = blob[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        extra = blob[off + 46 + nlen : off + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or loff == 0xFFFFFFFF:
            eoff = 0
            while eoff + 4 <= len(extra):
                eid, esz = struct.unpack_from("<HH", extra, eoff)
                payload = extra[eoff + 4 : eoff + 4 + esz]
                po = 0
                if eid == 1:
                    if usize == 0xFFFFFFFF and po + 8 <= len(payload):
                        usize = struct.unpack_from("<Q", payload, po)[0]
                        po += 8
                    if csize == 0xFFFFFFFF and po + 8 <= len(payload):
                        csize = struct.unpack_from("<Q", payload, po)[0]
                        po += 8
                    if loff == 0xFFFFFFFF and po + 8 <= len(payload):
                        loff = struct.unpack_from("<Q", payload, po)[0]
                        po += 8
                eoff += 4 + esz
        rows.append(
            {
                "name": name,
                "method": method,
                "csize": int(csize),
                "usize": int(usize),
                "loff": int(loff),
                "nlen": int(nlen),
            }
        )
        off += 46 + nlen + elen + clen
    return rows


def fetch_cd(cache: Path) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size == CD_SIZE:
        return cache.read_bytes()
    for src in (Path("/tmp/opxrd_cd.full.bin"), Path("/tmp/opxrd_cd.bin")):
        if src.exists() and src.stat().st_size == CD_SIZE:
            data = src.read_bytes()
            cache.write_bytes(data)
            return data
    print(f"Fetching zip central directory ({CD_SIZE} bytes) …", flush=True)
    data = http_range(OPXRD_URL, CD_OFF, CD_OFF + CD_SIZE - 1, timeout=300)
    cache.write_bytes(data)
    return data


def region_for(rows: list[dict], prefixes: tuple[str, ...], pad: int = 96) -> tuple[int, int, list[dict]]:
    sel = [
        r
        for r in rows
        if r["name"].endswith(".json") and r["name"].startswith(prefixes)
    ]
    if not sel:
        raise RuntimeError(f"no json members for {prefixes}")
    start = min(r["loff"] for r in sel)
    end = max(r["loff"] + 30 + r["nlen"] + pad + r["csize"] for r in sel)
    end = min(end, ZIP_SIZE)
    return start, end, sel


def download_region(start: int, end: int, part_dir: Path, label: str) -> bytes:
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    pos = start
    idx = 0
    while pos < end:
        chunk_end = min(pos + CHUNK, end) - 1
        ranges.append((idx, pos, chunk_end))
        pos = chunk_end + 1
        idx += 1
    print(f"{label}: bytes {start}-{end-1} ({end-start} B, {len(ranges)} chunks)", flush=True)

    def one(item):
        i, a, b = item
        part = part_dir / f"{label}.{i:04d}.part"
        expect = b - a + 1
        if part.exists() and part.stat().st_size == expect:
            return i, part.read_bytes()
        data = http_range(OPXRD_URL, a, b, timeout=180)
        part.write_bytes(data)
        print(f"  {label} chunk {i+1}/{len(ranges)} {a}-{b} ok", flush=True)
        return i, data

    pieces = [b""] * len(ranges)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, r) for r in ranges]
        for fut in as_completed(futs):
            i, data = fut.result()
            pieces[i] = data
            done += 1
            if done % 2 == 0 or done == len(ranges):
                got = sum(len(p) for p in pieces)
                print(f"  {label} assembled {done}/{len(ranges)} ({got} B)", flush=True)
    blob = b"".join(pieces)
    if len(blob) != end - start:
        raise RuntimeError(f"{label}: concatenated {len(blob)} != {end-start}")
    return blob


def extract_member(blob: bytes, base: int, row: dict) -> bytes:
    rel = row["loff"] - base
    if rel < 0 or rel + 30 > len(blob):
        raise RuntimeError(f"header out of range for {row['name']}")
    if blob[rel : rel + 4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {row['name']}")
    nlen, elen = struct.unpack_from("<HH", blob, rel + 26)
    data_off = rel + 30 + nlen + elen
    comp = blob[data_off : data_off + row["csize"]]
    if len(comp) != row["csize"]:
        raise RuntimeError(f"short payload for {row['name']}: {len(comp)}/{row['csize']}")
    if row["method"] == 8:
        return zlib.decompress(comp, -15)
    if row["method"] == 0:
        return comp
    raise RuntimeError(f"unsupported method {row['method']} for {row['name']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "public_exp_pxrd"
            / "raw"
            / "opxrd"
        ),
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cd_path = out_dir.parent / "opxrd_central_directory.bin"
    rows = parse_central_directory(fetch_cd(cd_path))
    print(f"central directory members: {len(rows)}", flush=True)

    extracted = 0
    failed = 0
    index = []
    for prefix in KEEP_PREFIXES:
        label = prefix.strip("/").replace("/", "_")
        start, end, sel = region_for(rows, (prefix,))
        part_dir = out_dir.parent / "opxrd_parts" / label
        blob = download_region(start, end, part_dir, label)
        print(f"Extracting {len(sel)} JSON files from {prefix} …", flush=True)
        for row in sel:
            dest = out_dir / row["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = extract_member(blob, start, row)
                dest.write_bytes(raw)
                extracted += 1
                index.append(
                    {
                        "name": row["name"],
                        "bytes": len(raw),
                        "zip_csize": row["csize"],
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"  FAIL {row['name']}: {exc}", flush=True)
        del blob

    summary = {
        "source": OPXRD_URL,
        "keep": list(KEEP_PREFIXES),
        "extracted": extracted,
        "failed": failed,
        "note": (
            "Only CNRS + HKUST-A: these are the opXRD subsets with atomic "
            "coordinates (Hollarek et al., Table 1). Unlabeled LBNL/INT/IKFT "
            "and lattice-only EMPA/USC are not downloaded."
        ),
    }
    (out_dir / "download_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "extracted_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    (out_dir / ".unpacked").write_text("labeled-only\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if extracted and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
