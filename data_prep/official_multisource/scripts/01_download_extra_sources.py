#!/usr/bin/env python
"""Download small or full raw exports for Alexandria, OQMD, and AFLOW.

This script is intentionally independent from the MP+WBM pipeline so it can be
run while the MP+WBM Slurm build is still producing LMDBs.

Usage:
  python 01_download_extra_sources.py --download-alexandria --alexandria-limit 100
  python 01_download_extra_sources.py --download-oqmd --oqmd-limit 100
  python 01_download_extra_sources.py --probe-aflow
  python 01_download_extra_sources.py --download-aflow-from-aurls --aflow-aurl-list aurls.txt
  python 01_download_extra_sources.py --download-aflow-catalog --aflow-catalog LIB2_RAW
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from typing import Any, TextIO

import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data_prep" / "raw" / "official_extra_sources"


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_row(f: TextIO, row: dict[str, Any]) -> None:
    f.write(json.dumps(row) + "\n")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def _count_skipped_offsets(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def _load_existing_aflow_aurls(jsonl_path: Path, failed_path: Path) -> set[str]:
    """Return AURLs already written or intentionally skipped for resumable AFLOW shards."""
    seen: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aurl = row.get("input_aurl") or row.get("aurl")
                if isinstance(aurl, str) and aurl:
                    seen.add(aurl)
    if failed_path.exists():
        with failed_path.open() as f:
            for line in f:
                aurl = line.split("\t", 1)[0].strip()
                if aurl:
                    seen.add(aurl)
    return seen


def download_alexandria(
    raw_dir: Path,
    dataset: str = "pbe",
    limit: int | None = None,
    page_limit: int = 100,
) -> Path:
    """Download Alexandria OPTIMADE structures as JSONL."""
    base = f"https://alexandria.icams.rub.de/{dataset}/v1/structures"
    out = raw_dir / (f"alexandria_{dataset}_sample.jsonl" if limit else f"alexandria_{dataset}.jsonl")
    skipped_offsets = raw_dir / f"alexandria_{dataset}_skipped_offsets.txt"
    session = requests.Session()
    session.headers.update({"user-agent": "pxrd_mof-official-extra-sources/0.1"})

    resume_count = _count_lines(out) if limit is None else 0
    resume_skipped = _count_skipped_offsets(skipped_offsets) if limit is None else 0
    count = resume_count
    offset = resume_count + resume_skipped
    mode = "a" if resume_count else "w"
    if resume_count or resume_skipped:
        print(
            f"Resuming Alexandria {dataset} from {resume_count} existing records "
            f"and {resume_skipped} skipped offsets; next offset={offset}"
        )
    print(f"Querying Alexandria {dataset} OPTIMADE -> {out}")
    with out.open(mode) as f, skipped_offsets.open("a") as skipped_f:
        while True:
            current_page_limit = min(page_limit, limit - count) if limit is not None else page_limit
            if current_page_limit <= 0:
                break
            resp = None
            while True:
                try:
                    resp = session.get(
                        base,
                        params={"page_limit": current_page_limit, "page_offset": offset},
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    if current_page_limit > 1:
                        current_page_limit = max(1, current_page_limit // 2)
                    print(
                        f"  Alexandria request error at offset={offset}: {exc!r}; "
                        f"retrying with page_limit={current_page_limit}"
                    )
                    sleep(30)
                    continue
                if resp.status_code >= 500 and current_page_limit > 1:
                    current_page_limit = max(1, current_page_limit // 2)
                    print(
                        f"  Alexandria HTTP {resp.status_code} at offset={offset}; "
                        f"retrying with page_limit={current_page_limit}"
                    )
                    sleep(5)
                    continue
                if resp.status_code >= 500 and current_page_limit == 1:
                    print(f"  Alexandria HTTP {resp.status_code} at single offset={offset}; skipping bad record")
                    skipped_f.write(f"{offset}\n")
                    skipped_f.flush()
                    offset += 1
                    sleep(1)
                    break
                resp.raise_for_status()
                break
            if resp is None:
                continue
            if resp.status_code >= 500 and current_page_limit == 1:
                continue
            try:
                payload = resp.json()
            except requests.RequestException as exc:
                print(f"  Alexandria JSON/read error at offset={offset}: {exc!r}; retrying")
                sleep(30)
                continue
            data = payload.get("data", [])
            if not data:
                break
            for row in data:
                if limit is not None and count >= limit:
                    break
                _write_row(f, row)
                count += 1
            offset += len(data)
            print(f"  downloaded {count} Alexandria records")
            if limit is not None and count >= limit:
                break
            if not payload.get("meta", {}).get("more_data_available", False):
                break
            sleep(0.2)
    print(f"Alexandria records written: {count}")
    return out


def download_oqmd(raw_dir: Path, limit: int | None = None, page_limit: int = 100) -> Path:
    """Download OQMD formationenergy endpoint records as JSONL."""
    out = raw_dir / ("oqmd_formationenergy_sample.jsonl" if limit else "oqmd_formationenergy.jsonl")
    session = requests.Session()
    session.headers.update({"user-agent": "pxrd_mof-official-extra-sources/0.1"})
    resume_count = _count_lines(out) if limit is None else 0
    count = resume_count
    offset = resume_count
    mode = "a" if resume_count else "w"
    if resume_count:
        print(f"Resuming OQMD from {resume_count} existing records")
    print(f"Querying OQMD formationenergy REST -> {out}")
    with out.open(mode) as f:
        while True:
            current_limit = min(page_limit, limit - count) if limit is not None else page_limit
            if current_limit <= 0:
                break
            url = "https://www.oqmd.org/oqmdapi/formationenergy"
            resp = None
            for attempt in range(10):
                try:
                    resp = session.get(
                        url,
                        params={"limit": current_limit, "offset": offset},
                        timeout=(10, 120),
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        wait_s = 10 * (attempt + 1)
                        print(
                            f"  OQMD HTTP {resp.status_code} at offset={offset}; "
                            f"retrying in {wait_s}s"
                        )
                        sleep(wait_s)
                        continue
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    wait_s = 10 * (attempt + 1)
                    print(
                        f"  OQMD request error at offset={offset}: {exc!r}; "
                        f"retrying in {wait_s}s"
                    )
                    sleep(wait_s)
            if resp is None:
                continue
            payload = None
            for attempt in range(10):
                try:
                    payload = resp.json()
                    break
                except (ValueError, requests.RequestException) as exc:
                    wait_s = 10 * (attempt + 1)
                    print(
                        f"  OQMD shard {shard_index} JSON error at offset={offset}: "
                        f"{exc!r}; retrying in {wait_s}s"
                    )
                    sleep(wait_s)
                    try:
                        resp = session.get(
                            url,
                            params={"limit": current_limit, "offset": offset},
                            timeout=(10, 120),
                        )
                        resp.raise_for_status()
                    except requests.RequestException as retry_exc:
                        print(
                            f"  OQMD shard {shard_index} retry request failed at "
                            f"offset={offset}: {retry_exc!r}"
                        )
            if payload is None:
                continue
            data = payload.get("data", [])
            for row in data:
                _write_row(f, row)
                count += 1
            offset += len(data)
            print(f"  downloaded {count} OQMD records; offset={offset}")
            if not data or len(data) < current_limit:
                break
            if limit is not None and count >= limit:
                break
            sleep(0.2)
    print(f"OQMD records written: {count}")
    return out


def _oqmd_get_total_records(timeout_s: int = 120) -> int:
    url = "https://www.oqmd.org/oqmdapi/formationenergy"
    resp = requests.get(url, params={"limit": 1, "offset": 0}, timeout=(10, timeout_s))
    resp.raise_for_status()
    return int(resp.json()["meta"]["data_available"])


def download_oqmd_shard(
    raw_dir: Path,
    shard_index: int,
    shard_count: int,
    page_limit: int = 100,
    total_records: int | None = None,
) -> Path:
    """Download one deterministic offset shard from OQMD.

    Each shard writes a separate JSONL to avoid concurrent writes to the same
    file.  The offsets are contiguous ranges, not interleaved, so each shard can
    resume by line count and the final merge is cheap.
    """
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid OQMD shard {shard_index}/{shard_count}")
    if total_records is None:
        total_records = _oqmd_get_total_records()
    shard_size = (total_records + shard_count - 1) // shard_count
    start = shard_index * shard_size
    end = min(total_records, start + shard_size)
    out = raw_dir / f"oqmd_formationenergy_shard_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    session = requests.Session()
    session.headers.update({"user-agent": "pxrd_mof-official-extra-sources/0.1"})

    resume_count = _count_lines(out)
    count = resume_count
    offset = start + resume_count
    mode = "a" if resume_count else "w"
    print(
        f"Querying OQMD shard {shard_index}/{shard_count}: "
        f"offset=[{start}, {end}) total={total_records}; resume={resume_count} -> {out}"
    )
    with out.open(mode) as f:
        while offset < end:
            current_limit = min(page_limit, end - offset)
            url = "https://www.oqmd.org/oqmdapi/formationenergy"
            resp = None
            for attempt in range(10):
                try:
                    resp = session.get(
                        url,
                        params={"limit": current_limit, "offset": offset},
                        timeout=(10, 120),
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        wait_s = 10 * (attempt + 1)
                        print(
                            f"  OQMD shard {shard_index} HTTP {resp.status_code} "
                            f"at offset={offset}; retrying in {wait_s}s"
                        )
                        sleep(wait_s)
                        continue
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    wait_s = 10 * (attempt + 1)
                    print(
                        f"  OQMD shard {shard_index} request error at offset={offset}: "
                        f"{exc!r}; retrying in {wait_s}s"
                    )
                    sleep(wait_s)
            if resp is None:
                continue
            payload = None
            for json_attempt in range(10):
                try:
                    payload = resp.json()
                    break
                except ValueError as exc:
                    wait_s = 10 * (json_attempt + 1)
                    print(
                        f"  OQMD shard {shard_index} JSON error at offset={offset}: "
                        f"{exc!r}; retrying in {wait_s}s"
                    )
                    sleep(wait_s)
                    try:
                        resp = session.get(
                            url,
                            params={"limit": current_limit, "offset": offset},
                            timeout=(10, 120),
                        )
                        if resp.status_code in (429, 500, 502, 503, 504):
                            print(
                                f"  OQMD shard {shard_index} HTTP {resp.status_code} "
                                f"during JSON retry at offset={offset}"
                            )
                            continue
                        resp.raise_for_status()
                    except requests.RequestException as retry_exc:
                        print(
                            f"  OQMD shard {shard_index} retry request failed at "
                            f"offset={offset}: {retry_exc!r}"
                        )
            if payload is None:
                continue
            data = payload.get("data", [])
            if not data:
                break
            for row in data:
                _write_row(f, row)
                count += 1
            offset += len(data)
            print(f"  shard {shard_index}: downloaded {count}; offset={offset}/{end}")
            f.flush()
            if len(data) < current_limit:
                break
            sleep(0.2)
    print(f"OQMD shard {shard_index}/{shard_count} records written: {count}")
    return out


def probe_aflow(raw_dir: Path, page_size: int = 1, timeout_s: int = 20) -> Path | None:
    """Probe AFLOW AFLUX with a tiny query and save the response if reachable."""
    query = (
        "Egap,enthalpy_formation_atom,species,positions_fractional,geometry,"
        "compound,aurl,auid,natoms,spacegroup_relax,paging(1,"
        f"{page_size}),format(json)"
    )
    url = "https://aflow.org/API/aflux/?" + query
    out = raw_dir / "aflow_probe.json"
    print(f"Probing AFLOW AFLUX -> {url}")
    try:
        resp = requests.get(
            url,
            timeout=(10, timeout_s),
            headers={"user-agent": "pxrd_mof-official-extra-sources/0.1"},
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"AFLOW probe failed or timed out: {exc!r}")
        return None
    out.write_text(resp.text)
    print(f"AFLOW probe response -> {out} ({out.stat().st_size / 1e3:.1f} KB)")
    return out


def _aflow_aurl_to_base_url(aurl: str) -> str:
    aurl = aurl.strip().strip('"')
    if not aurl:
        raise ValueError("Empty AFLOW AURL")
    if aurl.startswith("http://") or aurl.startswith("https://"):
        return aurl.rstrip("/")
    if ":" in aurl:
        host, path = aurl.split(":", 1)
        return f"https://{host}/{path.strip('/')}"
    return f"https://aflowlib.duke.edu/{aurl.strip('/')}"


def _aflow_timeout(timeout_s: int) -> tuple[int, int]:
    return (max(30, timeout_s), timeout_s)


def _aflow_get_json(session: requests.Session, base_url: str, keyword: str, timeout_s: int, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(
                f"{base_url}/?{keyword}",
                timeout=_aflow_timeout(timeout_s),
                headers={"accept": "application/json"},
            )
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            sleep(3 * (attempt + 1))
    raise last_exc or RuntimeError(f"Failed to fetch AFLOW keyword {keyword}")


def _aflow_get_structure_json(session: requests.Session, base_url: str, files: list[str], timeout_s: int) -> dict | None:
    candidates = [
        name
        for name in files
        if name.endswith("_structure_relax.json") or name.endswith("_structure_relax1.json")
    ]
    for name in candidates:
        for attempt in range(3):
            try:
                resp = session.get(f"{base_url}/{name}", timeout=_aflow_timeout(timeout_s))
                if resp.status_code == 404:
                    break
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == 2:
                    raise
                sleep(3 * (attempt + 1))
    return None


def _aflow_download_entry(session: requests.Session, aurl: str, timeout_s: int) -> dict[str, Any]:
    base_url = _aflow_aurl_to_base_url(aurl)
    keywords = [
        "aurl",
        "auid",
        "compound",
        "prototype",
        "species",
        "natoms",
        "geometry",
        "positions_fractional",
        "spacegroup_relax",
        "enthalpy_formation_atom",
        "Egap",
        "Egap_type",
        "dft_type",
        "code",
        "files",
    ]
    row: dict[str, Any] = {"input_aurl": aurl, "base_url": base_url}
    for keyword in keywords:
        row[keyword] = _aflow_get_json(session, base_url, keyword, timeout_s)
    files = row.get("files") or []
    if isinstance(files, list):
        row["structure_relax_json"] = _aflow_get_structure_json(session, base_url, files, timeout_s)
    return row


def _new_aflow_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"user-agent": "pxrd_mof-official-extra-sources/0.1"})
    return session


def _aflow_download_entry_threaded(aurl: str, timeout_s: int) -> dict[str, Any]:
    return _aflow_download_entry(_new_aflow_session(), aurl, timeout_s)


def download_aflow_from_aurls(
    raw_dir: Path,
    aurl_list: Path,
    limit: int | None = None,
    timeout_s: int = 30,
    sleep_s: float = 0.2,
    workers: int = 1,
) -> Path:
    """Download AFLOW entries by known AURL using per-entry REST API calls."""
    out = raw_dir / ("aflow_aurl_sample.jsonl" if limit else "aflow_aurl.jsonl")
    count = 0
    skipped = 0
    print(f"Downloading AFLOW entries from {aurl_list} -> {out}")
    aurls: list[str] = []
    with aurl_list.open() as src:
        for line in src:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            aurls.append(line)
            if limit is not None and len(aurls) >= limit:
                break
    with out.open("w") as dst:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(_aflow_download_entry_threaded, aurl, timeout_s): aurl for aurl in aurls}
            for future in as_completed(futures):
                aurl = futures[future]
                try:
                    row = future.result()
                    if row.get("structure_relax_json") is None and row.get("positions_fractional") is None:
                        skipped += 1
                        row["download_error"] = "missing structure fields"
                    _write_row(dst, row)
                    count += 1
                    if count % 10 == 0 or limit is not None:
                        print(f"  wrote {count} AFLOW entries; skipped={skipped}")
                except Exception as exc:
                    skipped += 1
                    print(f"  skip {aurl}: {exc!r}")
                sleep(sleep_s)
    print(f"AFLOW records written: {count}; skipped={skipped}")
    return out


def _aflow_get_entries(session: requests.Session, url: str, timeout_s: int, retries: int = 5) -> list[str]:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=_aflow_timeout(timeout_s))
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Expected list from {url}, got {type(data).__name__}")
            return [str(item) for item in data]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait_s = 10 * (attempt + 1)
            print(f"  AFLOW retry {attempt + 1}/{retries} for {url}: {exc!r}; sleeping {wait_s}s")
            sleep(wait_s)
    return []


def download_aflow_catalog(
    raw_dir: Path,
    catalog: str,
    system_list: Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
    timeout_s: int = 60,
    sleep_s: float = 0.2,
    workers: int = 1,
) -> Path:
    """Enumerate AFLOW systems/entries for one catalog shard and download records."""
    session = requests.Session()
    session.headers.update({"user-agent": "pxrd_mof-official-extra-sources/0.1"})
    catalog = catalog.strip("/")
    if system_list is not None:
        systems = [line.strip() for line in system_list.open() if line.strip() and not line.startswith("#")]
    else:
        systems = _aflow_get_entries(
            session,
            f"https://aflowlib.duke.edu/AFLOWDATA/{catalog}/?aflowlib_entries",
            timeout_s,
        )
    selected_systems = [system for i, system in enumerate(systems) if i % shard_count == shard_index]
    out = raw_dir / f"aflow_{catalog.lower()}_shard_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    aurl_out = raw_dir / f"aflow_{catalog.lower()}_shard_{shard_index:03d}_of_{shard_count:03d}.aurls.txt"
    failed_systems_out = raw_dir / f"aflow_{catalog.lower()}_shard_{shard_index:03d}_of_{shard_count:03d}.failed_systems.txt"
    failed_entries_out = raw_dir / f"aflow_{catalog.lower()}_shard_{shard_index:03d}_of_{shard_count:03d}.failed_entries.txt"
    seen_aurls = _load_existing_aflow_aurls(out, failed_entries_out) if limit is None else set()
    count = _count_lines(out) if limit is None else 0
    skipped = 0
    print(
        f"Downloading AFLOW catalog={catalog} shard={shard_index}/{shard_count}; "
        f"systems={len(selected_systems)} workers={workers} -> {out}"
    )
    if seen_aurls:
        print(f"Resuming AFLOW shard with {len(seen_aurls)} existing/failed AURLs and {count} records")
    with (
        out.open("a" if seen_aurls else "w") as dst,
        aurl_out.open("a" if seen_aurls else "w") as aurl_dst,
        failed_systems_out.open("a" if seen_aurls else "w") as failed_systems,
        failed_entries_out.open("a" if seen_aurls else "w") as failed_entries,
    ):
        for system_i, system in enumerate(selected_systems, 1):
            entries_url = f"https://aflowlib.duke.edu/AFLOWDATA/{catalog}/{system}/?aflowlib_entries"
            try:
                entries = _aflow_get_entries(session, entries_url, timeout_s)
            except Exception as exc:
                skipped += 1
                failed_systems.write(f"{system}\t{exc!r}\n")
                failed_systems.flush()
                print(f"  skip system {system}: {exc!r}")
                continue
            print(f"  system {system_i}/{len(selected_systems)} {system}: {len(entries)} entries")
            aurls = []
            for entry in entries:
                aurl = f"aflowlib.duke.edu:AFLOWDATA/{catalog}/{system}/{entry}"
                if aurl in seen_aurls:
                    continue
                aurl_dst.write(aurl + "\n")
                if limit is None or count + len(aurls) < limit:
                    aurls.append(aurl)
            aurl_dst.flush()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(_aflow_download_entry_threaded, aurl, timeout_s): aurl for aurl in aurls}
                for future in as_completed(futures):
                    aurl = futures[future]
                    if limit is not None and count >= limit:
                        break
                    try:
                        row = future.result()
                    except Exception as exc:
                        skipped += 1
                        failed_entries.write(f"{aurl}\t{exc!r}\n")
                        failed_entries.flush()
                        print(f"    skip {aurl}: {exc!r}")
                        continue
                    if row.get("structure_relax_json") is None and row.get("positions_fractional") is None:
                        skipped += 1
                        row["download_error"] = "missing structure fields"
                    _write_row(dst, row)
                    count += 1
                    if count % 10 == 0:
                        print(f"    wrote {count} AFLOW entries; skipped={skipped}")
                    sleep(sleep_s)
            dst.flush()
            if limit is not None and count >= limit:
                break
    print(f"AFLOW catalog records written: {count}; skipped={skipped}; aurls -> {aurl_out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--download-alexandria", action="store_true")
    parser.add_argument("--alexandria-dataset", default="pbe", choices=["pbe", "pbesol"])
    parser.add_argument("--alexandria-limit", type=int, default=None)
    parser.add_argument("--alexandria-page-limit", type=int, default=100)
    parser.add_argument("--download-oqmd", action="store_true")
    parser.add_argument("--download-oqmd-shard", action="store_true")
    parser.add_argument("--oqmd-limit", type=int, default=None)
    parser.add_argument("--oqmd-page-limit", type=int, default=100)
    parser.add_argument("--oqmd-shard-index", type=int, default=0)
    parser.add_argument("--oqmd-shard-count", type=int, default=1)
    parser.add_argument("--oqmd-total-records", type=int, default=None)
    parser.add_argument("--probe-aflow", action="store_true")
    parser.add_argument("--download-aflow-from-aurls", action="store_true")
    parser.add_argument("--download-aflow-catalog", action="store_true")
    parser.add_argument("--aflow-catalog", default="LIB2_RAW")
    parser.add_argument("--aflow-system-list", default=None)
    parser.add_argument("--aflow-shard-index", type=int, default=0)
    parser.add_argument("--aflow-shard-count", type=int, default=1)
    parser.add_argument("--aflow-aurl-list", default=None)
    parser.add_argument("--aflow-limit", type=int, default=None)
    parser.add_argument("--aflow-timeout-s", type=int, default=20)
    parser.add_argument("--aflow-sleep-s", type=float, default=0.2)
    parser.add_argument("--aflow-workers", type=int, default=1)
    args = parser.parse_args(argv)

    if not (
        args.download_alexandria
        or args.download_oqmd
        or args.download_oqmd_shard
        or args.probe_aflow
        or args.download_aflow_from_aurls
        or args.download_aflow_catalog
    ):
        parser.error(
            "Choose at least one of --download-alexandria / --download-oqmd / "
            "--download-oqmd-shard / "
            "--probe-aflow / --download-aflow-from-aurls / --download-aflow-catalog"
        )

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest["raw_dir"] = str(raw_dir)

    if args.download_alexandria:
        path = download_alexandria(
            raw_dir,
            dataset=args.alexandria_dataset,
            limit=args.alexandria_limit,
            page_limit=args.alexandria_page_limit,
        )
        manifest[f"alexandria_{args.alexandria_dataset}_jsonl"] = str(path)

    if args.download_oqmd:
        path = download_oqmd(raw_dir, limit=args.oqmd_limit, page_limit=args.oqmd_page_limit)
        manifest["oqmd_formationenergy_jsonl"] = str(path)

    if args.download_oqmd_shard:
        path = download_oqmd_shard(
            raw_dir,
            shard_index=args.oqmd_shard_index,
            shard_count=args.oqmd_shard_count,
            page_limit=args.oqmd_page_limit,
            total_records=args.oqmd_total_records,
        )
        manifest[f"oqmd_formationenergy_shard_{args.oqmd_shard_index}_of_{args.oqmd_shard_count}_jsonl"] = str(path)

    if args.probe_aflow:
        path = probe_aflow(raw_dir, timeout_s=args.aflow_timeout_s)
        manifest["aflow_probe_json"] = str(path) if path else None

    if args.download_aflow_from_aurls:
        if not args.aflow_aurl_list:
            raise SystemExit("--aflow-aurl-list is required with --download-aflow-from-aurls")
        path = download_aflow_from_aurls(
            raw_dir,
            Path(args.aflow_aurl_list),
            limit=args.aflow_limit,
            timeout_s=args.aflow_timeout_s,
            sleep_s=args.aflow_sleep_s,
            workers=args.aflow_workers,
        )
        manifest["aflow_aurl_jsonl"] = str(path)

    if args.download_aflow_catalog:
        path = download_aflow_catalog(
            raw_dir,
            catalog=args.aflow_catalog,
            system_list=Path(args.aflow_system_list) if args.aflow_system_list else None,
            shard_index=args.aflow_shard_index,
            shard_count=args.aflow_shard_count,
            limit=args.aflow_limit,
            timeout_s=args.aflow_timeout_s,
            sleep_s=args.aflow_sleep_s,
            workers=args.aflow_workers,
        )
        manifest[f"aflow_{args.aflow_catalog.lower()}_shard_{args.aflow_shard_index}_jsonl"] = str(path)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

