#!/usr/bin/env python3
"""
Module: dataload.py

Loads and preprocesses AIPP data for grouping. Responsibilities:
  1. Load lcr_*.dat files, split single/ambiguous/multi-value lines.
  2. Write initial reports: statistics, splitting, summary.
  3. Fetch UniProt and PDB-CHAIN sequences, validating that every ROI is Cys.
  4. Extract a subsequence window (default 21) around each ROI (mapped for
     PDB).
  5. Write FASTA of accepted sequences, a FASTA of unacceptable sequences,
     and a failure log.
  6. Return in-memory list of “accepted” records, each annotated with:
       • sequence (string)
       • mapped_roi (1-indexed into sequence)
       • window_seq (substring around mapped_roi, length ≤ window)
       • original metadata: uid, roi, source, note, etc.

Usage from `aipp-group-distiller`:

    from dataload import load_dataset

    records = load_dataset(
        data_dir="path/to/dat_dir",
        report_dir="path/to/reports",
        window=21,
        ignore="SRC1,SRC2",
        threads=8,
        nostrict=False
    )
"""

import os
import glob
import argparse
import re
import json
import requests
import time
import random
from collections import Counter, defaultdict
from colorama import init as _clr_init, Fore as C, Style as S
from colorama import Fore, Style
from tqdm import tqdm
from multiprocessing import Pool
import pdbdr  # ensure pdbdr is installed: pip install pdbdr

import aipp.distill as _distill
import aipp.report as report

# ──────────────────────────────────────────────────────────────────────────
# Constants & Globals
# ──────────────────────────────────────────────────────────────────────────
WIDTH = 80
PDB_CACHE_DIR = "pdb_cache"
os.makedirs(PDB_CACHE_DIR, exist_ok=True)

# Pattern for PDB-like UID: 4 alphanumerics, hyphen, then chain
PDB_UID_REGEX = re.compile(r"^[A-Za-z0-9]{4}-[A-Za-z0-9]+$")


def _banner(txt: str, colour=C.CYAN, fill="─") -> None:
    """
    Print a centered banner with a colored line above/below.
    """
    msg = f" {txt.strip()} "
    pad = max(WIDTH - len(msg), 0)
    left = pad // 2
    right = pad - left
    print(colour + fill * left + msg + fill * right + S.RESET_ALL)


def _info(txt: str, colour=C.GREEN) -> None:
    """
    Print an informational line (colored, fixed width).
    """
    print(colour + txt.ljust(WIDTH) + S.RESET_ALL)


def load_and_split_data(directory, ignore_sources=None):
    """
    Load all 'lcr_*.dat' files from `directory`, skipping any whose source
    code is in ignore_sources.

    For each line:
      - Classify as single-value, ambiguous, or multi-value.
      - Split ambiguous (1 UID, >1 ROIs) → 1 record per ROI.
      - Split multi-value (multiple UIDs) → 1 record per (UID, ROI):
          • If #UID == #ROI, pair index-wise.
          • If single ROI & multiple UIDs, replicate ROI for each UID.
          • Else fallback to Cartesian product.

    Returns:
      all_records: list of dicts with keys:
        'uid','roi','R','exp_thr','exp_bin','source','note','mw',
        'orig_ambiguous','orig_multi_value','og_uid'
      counts: dict of raw-line counts
      splitting_events: list describing each split action
    """
    pattern = os.path.join(directory, "lcr_*.dat")
    files = sorted(glob.glob(pattern))

    # Filter out files whose source code is in ignore_sources
    if ignore_sources:
        kept = []
        for fn in files:
            base = os.path.basename(fn)
            if base.startswith("lcr_") and base.endswith(".dat"):
                code = base[len("lcr_"):-len(".dat")]
                if code in ignore_sources:
                    continue
            kept.append(fn)
        files = kept

    all_records = []
    splitting_events = []
    orig_single = 0
    orig_ambiguous = 0
    orig_multi = 0

    for filename in tqdm(files, desc="Loading files", ncols=80):
        source_code = os.path.basename(filename)[len("lcr_"):-len(".dat")]
        with open(filename, "r") as f:
            lines = f.readlines()

        # Skip header if present
        start_idx = 0
        if lines and lines[0].lower().startswith("uid"):
            start_idx = 1

        for raw_line in tqdm(
            lines[start_idx:],
            desc=f"Processing {source_code}",
            ncols=80,
            leave=False
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            uid_field = parts[0]
            roi_field = parts[1]
            R_field = parts[2]
            exp_thr_field = parts[3]
            exp_bin_field = parts[4]
            source_field = parts[5]
            note_field = parts[6]
            mw_field = parts[7] if len(parts) >= 8 else None

            # Ensure source_field matches filename's source_code
            if source_field.strip() != source_code:
                source_field = source_code

            # Parse numeric fields
            try:
                R_value = float(R_field)
                exp_thr_value = float(exp_thr_field)
                exp_bin_value = exp_bin_field.strip().upper() == "TRUE"
            except Exception:
                continue

            # Split UID and ROI by semicolons
            uid_vals = [x.strip() for x in uid_field.split(";") if x.strip()]
            roi_vals = [x.strip() for x in roi_field.split(";") if x.strip()]

            num_uid = len(uid_vals)
            num_roi = len(roi_vals)

            # Safe int parse for ROI
            def parse_int_or_skip(val):
                try:
                    return int(val)
                except ValueError:
                    return None

            parsed_rois = [parse_int_or_skip(rv) for rv in roi_vals]
            parsed_rois = [r for r in parsed_rois if r is not None]
            if not parsed_rois:
                continue

            # ── classify & split ─────────────────────────────────────────
            if num_uid == 1 and num_roi == 1:
                # Single-value
                orig_single += 1
                uid0 = uid_vals[0]
                roi_int = parsed_rois[0]
                record = {
                    "uid": uid0,
                    "roi": roi_int,
                    "R": R_value,
                    "exp_thr": exp_thr_value,
                    "exp_bin": exp_bin_value,
                    "source": source_field.strip(),
                    "note": note_field.strip(),
                    "mw": (
                        float(mw_field)
                        if (mw_field and mw_field.replace(".", "", 1).isdigit())
                        else None
                    ),
                    "orig_ambiguous": False,
                    "orig_multi_value": False,
                    "og_uid": uid_field.strip(),
                }
                all_records.append(record)

            elif num_uid == 1 and num_roi > 1:
                # Ambiguous (1 UID, multiple ROIs)
                orig_ambiguous += 1
                uid0 = uid_vals[0]
                splits = []
                for roi_int in parsed_rois:
                    splits.append((uid0, roi_int))
                    record = {
                        "uid": uid0,
                        "roi": roi_int,
                        "R": R_value,
                        "exp_thr": exp_thr_value,
                        "exp_bin": exp_bin_value,
                        "source": source_field.strip(),
                        "note": f"{note_field.strip()} (am)",
                        "mw": (
                            float(mw_field)
                            if (
                                mw_field and
                                mw_field.replace(".", "", 1).isdigit()
                            ) else None
                        ),
                        "orig_ambiguous": True,
                        "orig_multi_value": False,
                        "og_uid": uid_field.strip(),
                    }
                    all_records.append(record)
                splitting_events.append({
                    "file": filename,
                    "orig_line": line,
                    "type": "ambiguous",
                    "orig_uid_field": uid_field.strip(),
                    "orig_roi_field": roi_field.strip(),
                    "splits": splits
                })

            elif num_uid > 1:
                # Multi-value (multiple UIDs)
                orig_multi += 1
                splits = []
                if num_uid == num_roi:
                    # Pair index-wise
                    for i in range(num_uid):
                        uid0 = uid_vals[i]
                        roi_int = parsed_rois[i]
                        splits.append((uid0, roi_int))
                        record = {
                            "uid": uid0,
                            "roi": roi_int,
                            "R": R_value,
                            "exp_thr": exp_thr_value,
                            "exp_bin": exp_bin_value,
                            "source": source_field.strip(),
                            "note": f"{note_field.strip()} (mv)",
                            "mw": (
                                float(mw_field)
                                if (
                                    mw_field and
                                    mw_field.replace(".", "", 1).isdigit()
                                ) else None
                            ),
                            "orig_ambiguous": False,
                            "orig_multi_value": True,
                            "og_uid": uid_field.strip(),
                        }
                        all_records.append(record)

                elif num_roi == 1:
                    # Single ROI for multiple UIDs
                    roi_int = parsed_rois[0]
                    for uid0 in uid_vals:
                        splits.append((uid0, roi_int))
                        record = {
                            "uid": uid0,
                            "roi": roi_int,
                            "R": R_value,
                            "exp_thr": exp_thr_value,
                            "exp_bin": exp_bin_value,
                            "source": source_field.strip(),
                            "note": f"{note_field.strip()} (mv)",
                            "mw": (
                                float(mw_field)
                                if (
                                    mw_field and
                                    mw_field.replace(".", "", 1).isdigit()
                                ) else None
                            ),
                            "orig_ambiguous": False,
                            "orig_multi_value": True,
                            "og_uid": uid_field.strip(),
                        }
                        all_records.append(record)

                else:
                    # Fallback: Cartesian product
                    for uid0 in uid_vals:
                        for roi_int in parsed_rois:
                            splits.append((uid0, roi_int))
                            record = {
                                "uid": uid0,
                                "roi": roi_int,
                                "R": R_value,
                                "exp_thr": exp_thr_value,
                                "exp_bin": exp_bin_value,
                                "source": source_field.strip(),
                                "note": f"{note_field.strip()} (mv)",
                                "mw": (
                                    float(mw_field)
                                    if (
                                        mw_field and
                                        mw_field.replace(".", "", 1).isdigit()
                                    ) else None
                                ),
                                "orig_ambiguous": False,
                                "orig_multi_value": True,
                                "og_uid": uid_field.strip(),
                            }
                            all_records.append(record)

                splitting_events.append({
                    "file": filename,
                    "orig_line": line,
                    "type": "multi",
                    "orig_uid_field": uid_field.strip(),
                    "orig_roi_field": roi_field.strip(),
                    "splits": splits
                })

            else:
                # Should not occur
                continue

    counts = {
        "orig_single": orig_single,
        "orig_ambiguous": orig_ambiguous,
        "orig_multi": orig_multi,
    }
    return all_records, counts, splitting_events


def write_splitting_report(splitting_events, report_dir):
    """
    Write a detailed report describing each ambiguous or multi-value split.
    """
    report_path = os.path.join(report_dir, "splitting_report.txt")
    with open(report_path, "w") as f:
        f.write("SPLITTING REPORT\n")
        f.write("================\n\n")
        if not splitting_events:
            f.write("No ambiguous or multi-value entries were found.\n")
            return

        for event in splitting_events:
            f.write(f"File      : {os.path.basename(event['file'])}\n")
            f.write(f"Orig Line : {event['orig_line']}\n")
            f.write(f"Type      : {event['type'].upper()}\n")
            f.write(f"UID Field : {event['orig_uid_field']}\n")
            f.write(f"ROI Field : {event['orig_roi_field']}\n")
            f.write("Splits     \n")
            for (uid, roi) in event["splits"]:
                f.write(f"  - UID: {uid}, ROI: {roi}\n")
            f.write("\n")
    _info(f"Splitting report written: {report_path}", C.CYAN)


def write_custom_summary(total_records, counts, report_dir):
    """
    Produce a short, readable summary of split counts, per-source stats,
    and UID+ROI uniqueness metrics. Helpful for quick QA.
    """
    total_after_split = len(total_records)
    orig_single_count = counts["orig_single"]
    orig_ambiguous_count = counts["orig_ambiguous"]
    orig_multi_count = counts["orig_multi"]

    # Split counts
    split_from_ambig = sum(1 for r in total_records if r.get("orig_ambiguous"))
    split_from_multi = sum(1 for r in total_records if r.get("orig_multi_value"))

    # Records per source (post-splitting)
    per_source_counts = Counter(r["source"] for r in total_records)

    # Protein-level: per source, distinct UIDs
    proteins_per_source = defaultdict(set)
    for r in total_records:
        proteins_per_source[r["source"]].add(r["uid"])
    proteins_count_per_source = {
        src: len(uids) for src, uids in proteins_per_source.items()
    }

    # Overall unique proteins
    unique_proteins = set(r["uid"] for r in total_records)
    total_unique_proteins = len(unique_proteins)

    # UID+ROI → sources mapping
    uroi_to_sources = defaultdict(set)
    for r in total_records:
        key = (r["uid"], r["roi"])
        uroi_to_sources[key].add(r["source"])

    distinct_pairs = len(uroi_to_sources)
    shared_pairs_count = sum(
        1 for sources in uroi_to_sources.values() if len(sources) > 1
    )
    exclusive_pairs_count = sum(
        1 for sources in uroi_to_sources.values() if len(sources) == 1
    )

    # Unique contributions per source
    unique_contrib_per_source = Counter()
    for (uid, roi), sources in uroi_to_sources.items():
        if len(sources) == 1:
            only_src = next(iter(sources))
            unique_contrib_per_source[only_src] += 1

    summary_path = os.path.join(report_dir, "initial_summary.txt")
    with open(summary_path, "w") as f:
        f.write("ORIGINAL RAW-LINE COUNTS\n")
        f.write("------------------------\n")
        f.write(f"  Single-value lines : {orig_single_count}\n")
        f.write(f"  Ambiguous lines    : {orig_ambiguous_count}\n")
        f.write(f"  Multi-value lines  : {orig_multi_count}\n\n")

        f.write("POST-SPLITTING COUNTS\n")
        f.write("---------------------\n")
        f.write(
            "  Total records after splitting            : "
            f"{total_after_split}\n"
        )
        f.write(
            "    Split from ambiguous originals         : "
            f"{split_from_ambig}\n"
        )
        f.write(
            "    Split from multi-value originals       : "
            f"{split_from_multi}\n\n"
        )

        f.write("RECORDS PER SOURCE (AFTER SPLITTING)\n")
        f.write("------------------------------------\n")
        for src, cnt in per_source_counts.most_common():
            f.write(f"  {src:<15s} {cnt}\n")
        f.write("\n")

        f.write("PROTEIN-LEVEL METRICS\n")
        f.write("---------------------\n")
        f.write(
            "  Total unique proteins (all sources)      : "
            f"{total_unique_proteins}\n"
        )
        f.write("  Unique proteins per source:\n")
        for src in sorted(per_source_counts.keys()):
            prot_cnt = proteins_count_per_source.get(src, 0)
            f.write(f"    {src:<15s} {prot_cnt}\n")
        f.write("\n")

        f.write("UNIQUE (UID,ROI) CONTRIBUTIONS PER SOURCE\n")
        f.write("-----------------------------------------\n")
        for src in sorted(per_source_counts.keys()):
            unique_cnt = unique_contrib_per_source.get(src, 0)
            f.write(f"  {src:<15s} {unique_cnt}\n")
        f.write("\n")

        f.write("OVERALL UID+ROI STATISTICS\n")
        f.write("--------------------------\n")
        f.write(
            "  Total distinct (UID,ROI) pairs          : "
            f"{distinct_pairs}\n"
        )
        f.write(
            "  Pairs shared across multiple sources     : "
            f"{shared_pairs_count}\n"
        )
        f.write(
            "  Pairs unique to exactly one source      : "
            f"{exclusive_pairs_count}\n"
        )

    _info(f"Custom summary written: {summary_path}", C.CYAN)


def get_pdb_file(pdb_id, cache_dir=PDB_CACHE_DIR):
    """
    Download PDB into cache. Try RCSB, then PDBe. Return path or None.
    """
    pdb_id_uc = pdb_id.upper()
    out_path = os.path.join(cache_dir, f"{pdb_id_uc}.pdb")
    if os.path.isfile(out_path):
        return out_path

    # Try RCSB
    url1 = f"https://files.rcsb.org/download/{pdb_id_uc}.pdb"
    try:
        r = requests.get(url1, timeout=10)
        if r.status_code == 200 and r.text.startswith("HEADER"):
            with open(out_path, "w") as f:
                f.write(r.text)
            return out_path
    except Exception:
        pass

    # Fallback to PDBe
    url2 = (
        "https://www.ebi.ac.uk/pdbe/entry-files/download/"
        f"{pdb_id_uc.lower()}.pdb"
    )
    try:
        r2 = requests.get(url2, timeout=10)
        if r2.status_code == 200 and r2.text.startswith("HEADER"):
            with open(out_path, "w") as f:
                f.write(r2.text)
            return out_path
    except Exception:
        pass

    return None


def fetch_pdb_sequence_and_map(uid, rois):
    """
    For uid "PDBID-CHAIN" and a list of 1-indexed `rois`:
      • Download PDB, extract chain via pdbdr.pdb_to_tensor
      • Reject any sequence with '?'
      • Validate each ROI maps to a Cys
      • Map each original ROI → its 1-indexed position in full_seq

    Returns (full_seq, roi_map, None) on success, else (None, None, error).
    """
    pdb_id, chain_id = uid.split("-", 1)
    chain_id = chain_id.upper()
    pdb_file = get_pdb_file(pdb_id, PDB_CACHE_DIR)
    if not pdb_file:
        return None, None, f"Failed to download PDB {pdb_id}"

    try:
        ret = pdbdr.pdb_to_tensor(pdb_file, chain_id)
        # New (5-tuple): (atom, full_seq, masked_seq, seq_pos_map, chain_id)
        try:
            _atom, full_seq, _masked, seq_pos_map, _cid = ret
        except ValueError:
            # Legacy (4-tuple): (atom, full_seq, coords, seq_pos_map)
            _atom, full_seq, _coords, seq_pos_map = ret
        if not full_seq:
            return None, None, f"No sequence extracted for {uid}"
    except Exception as e:
        return None, None, f"pdbdr error for {uid}: {e}"

    # Reject unknowns
    if "?" in full_seq:
        return None, None, (
            f"Sequence contains unknown residue '?' for {uid}\n{full_seq}\n"
        )

    # Build inverse map: (resSeq, insertionCode) → sequential index (1-based)
    inv = {}
    for idx, pdb_key in seq_pos_map.items():
        inv[pdb_key] = idx  # idx is 1-based

    roi_map = {}
    for roi in rois:
        key = (roi, "")
        if key not in inv:
            return None, None, f"ROI {roi} not found in PDB mapping for {uid}"
        seq_idx = inv[key]
        if seq_idx < 1 or seq_idx > len(full_seq):
            return None, None, (
                f"Mapped index {seq_idx} out of range (length={len(full_seq)}) "
                f"for {uid}"
            )
        if full_seq[seq_idx - 1] != "C":
            return None, None, f"Residue at index {seq_idx} is not CYS for {uid}"
        roi_map[roi] = seq_idx

    return full_seq, roi_map, None


def _fetch_uniprot_fasta(uid, max_retries=5):
    """
    Robust UniProt FASTA fetch with retries and dual endpoints. Returns
    (seq, None) on success or (None, "error") on failure.
    """
    endpoints = [
        f"https://rest.uniprot.org/uniprotkb/{uid}.fasta",
        f"https://www.uniprot.org/uniprot/{uid}.fasta",
    ]
    headers = {
        "Accept": "text/x-fasta",
        "User-Agent": "AIPP/1.0 (+https://example.invalid) Python-requests",
    }
    for attempt in range(1, max_retries + 1):
        for url in endpoints:
            try:
                r = requests.get(url, headers=headers, timeout=(5, 30))
                if r.status_code in (429, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_s = int(retry_after)
                    else:
                        sleep_s = min(
                            30, (1.2 ** attempt) + random.uniform(0, 0.5)
                        )
                    time.sleep(sleep_s)
                    continue
                if r.status_code == 200:
                    lines = r.text.splitlines()
                    seq = "".join(
                        line for line in lines if not line.startswith(">")
                    ).strip()
                    if not seq:
                        return None, f"Empty FASTA for {uid}"
                    return seq, None
                if 400 <= r.status_code < 500:
                    return None, f"HTTP {r.status_code} for {uid}"
            except requests.Timeout:
                pass
            except Exception as e:
                last_err = str(e)  # noqa: F841 (kept for debug)
        time.sleep(min(30, (1.5 ** attempt) + random.uniform(0, 0.5)))
    return None, f"Request error for {uid}: exhausted retries"


def fetch_uniprot_sequence_and_map(uid, rois):
    """
    Fetch UniProt sequence and validate that each ROI is Cys. Reject any
    sequence containing 'X'. Returns (seq, roi_map, None) or error triple.
    """
    # Try local cache first
    if _distill._UNIPROT_LOCAL_DB is not None:
        full_seq = _distill._UNIPROT_LOCAL_DB.get(uid)
        if full_seq is not None:
            length = len(full_seq)
            roi_map = {}
            for roi in rois:
                if roi < 1 or roi > length:
                    return None, None, (
                        f"ROI {roi} out of range (1–{length}) for UniProt {uid}"
                    )
                if full_seq[roi - 1] != "C":
                    return None, None, (
                        f"Residue at ROI {roi} is not CYS for {uid}"
                    )
                roi_map[roi] = roi
            return full_seq, roi_map, None

    # REST with retries
    full_seq, err = _fetch_uniprot_fasta(uid)
    if err:
        return None, None, f"{err}"

    # Reject any 'X' in the sequence
    if "X" in full_seq:
        return None, None, (
            f"Sequence contains non-canonical residue 'X' for {uid}"
        )

    length = len(full_seq)
    roi_map = {}
    for roi in rois:
        if roi < 1 or roi > length:
            return None, None, (
                f"ROI {roi} out of range (1–{length}) for UniProt {uid}"
            )
        if full_seq[roi - 1] != "C":
            return None, None, f"Residue at ROI {roi} is not CYS for {uid}"
        roi_map[roi] = roi

    return full_seq, roi_map, None


def is_pdb_uid(uid):
    """
    True iff uid matches the PDB-CHAIN pattern.
    """
    return bool(PDB_UID_REGEX.match(uid))


def compute_window(full_seq, mapped_idx, window):
    """
    Extract subsequence of length up to `window` centered at `mapped_idx`
    (1-based). Near edges, take as many as available.
    """
    half = window // 2
    seq_len = len(full_seq)
    start = max(mapped_idx - half, 1)
    end = min(mapped_idx + half, seq_len)
    return full_seq[start - 1:end]


def fetch_sequence_worker(args):
    """
    Worker to fetch sequence + ROI map for a single UID.
    args = (uid, list_of_rois)
    Returns (uid, seq/None, roi_map/None, err/None).
    """
    uid, rois = args
    if is_pdb_uid(uid):
        return uid, *fetch_pdb_sequence_and_map(uid, rois)
    else:
        return uid, *fetch_uniprot_sequence_and_map(uid, rois)


def fetch_sequence_only(uid):
    """
    Fetch only the raw sequence (no ROI validation). Returns
    (uid, seq, err_or_None).
    """
    if is_pdb_uid(uid):
        full_seq, _, err = fetch_pdb_sequence_and_map(uid, [])
        return uid, full_seq, err
    else:
        full_seq, _, err = fetch_uniprot_sequence_and_map(uid, [])
        return uid, full_seq, err


def load_dataset(
    data_dir,
    report_dir,
    window=21,
    ignore=None,
    threads=8,
    nostrict=None
):
    """
    Master function to load and preprocess the dataset.

    Args:
      data_dir   : directory containing lcr_*.dat
      report_dir : where to write all reports & FASTA files
      window     : odd integer window size (default 21)
      ignore     : comma-separated source codes to ignore (or None)
      threads    : thread count for sequence fetching
      nostrict   : if True, drop only individual ROI mismatches (not UID)

    Returns:
      accepted_records: list of dicts with keys:
        uid, roi, R, exp_thr, exp_bin, source, note, mw, og_uid,
        sequence, mapped_roi, window_seq
      total_post_split: int (# post-split records pre-validation)
      accepted_pairs  : int (# unique (UID, ROI) among accepted)
    """
    if window % 2 == 0:
        raise ValueError("`window` must be an odd integer")

    _clr_init(autoreset=True)

    # Parse ignore into set
    ignore_set = set()
    if ignore:
        ignore_set = {t.strip() for t in ignore.split(",") if t.strip()}
        _info(
            f"Ignoring sources: {', '.join(sorted(ignore_set))}",
            C.YELLOW
        )

    # ── 1) Load & split raw records ───────────────────────────────────
    total_records, counts, splitting_events = load_and_split_data(
        data_dir,
        ignore_sources=ignore_set
    )
    _info(f"Orig single-value lines      : {counts['orig_single']}", C.GREEN)
    _info(f"Orig ambiguous lines         : {counts['orig_ambiguous']}", C.GREEN)
    _info(f"Orig multi-value lines       : {counts['orig_multi']}", C.GREEN)
    _info(f"Total records after splitting: {len(total_records)}", C.GREEN)

    # ── 1a) Per-source split statistics (colored table) ───────────────
    from collections import defaultdict  # local aliasing is fine

    per_source = defaultdict(list)
    for r in total_records:
        per_source[r["source"]].append(r)

    print("\n@Loading source data and breaking down...")

    GRAY = Fore.LIGHTBLACK_EX

    header_top = (
        "        " + "==========RECORDS=========" + "." +
        "=========ROI==========" + "." + "=========UID========="
    )
    header_cols = (
        "        " + " total     svr   mvr   amb" +
        "|" + "  total    dup    new " +
        "|" + " total    dup    new "
    )
    underline = (
        "        " + "==========================" +
        "|" + "=======================" +
        "|" + "======================="
    )

    def colorize_line(raw: str) -> str:
        out = []
        for ch in raw:
            if ch in ("=", "|", ":"):
                out.append(f"{GRAY}{ch}{Style.RESET_ALL}")
            else:
                out.append(ch)
        return "".join(out)

    print(colorize_line(header_top))
    print(colorize_line(header_cols))
    print(colorize_line(underline))

    tot_rec = tot_svr = tot_mvr = tot_amb = 0
    tot_roi_tot = tot_roi_dup = tot_roi_new = 0
    tot_uid_tot = tot_uid_dup = tot_uid_new = 0

    seen_roi_pairs = set()
    seen_uids = set()

    for src, recs in sorted(per_source.items()):
        rec_total = len(recs)
        ambig_cnt = sum(1 for r in recs if r.get("orig_ambiguous"))
        multi_cnt = sum(1 for r in recs if r.get("orig_multi_value"))
        single_cnt = rec_total - ambig_cnt - multi_cnt

        roi_pairs = {(r["uid"], r["roi"]) for r in recs}
        roi_total = len(roi_pairs)
        roi_dup = len(roi_pairs & seen_roi_pairs)
        roi_new = roi_total - roi_dup

        uids = {r["uid"] for r in recs}
        uid_total = len(uids)
        uid_dup = len(uids & seen_uids)
        uid_new = uid_total - uid_dup

        seen_roi_pairs |= roi_pairs
        seen_uids |= uids

        tot_rec += rec_total
        tot_svr += single_cnt
        tot_mvr += multi_cnt
        tot_amb += ambig_cnt
        tot_roi_tot += roi_total
        tot_roi_dup += roi_dup
        tot_roi_new += roi_new
        tot_uid_tot += uid_total
        tot_uid_dup += uid_dup
        tot_uid_new += uid_new

        row = (
            f"{src}:".rjust(7) + "  " +
            f"{rec_total:>5}" + "   " +
            f"{single_cnt:>5}" + "   " +
            f"{multi_cnt:>4}" + "   " +
            f"{ambig_cnt:>3}" + GRAY + "|" + Style.RESET_ALL + "  " +
            f"{roi_total:>5}" + "     " +
            f"{roi_dup:>3}" + "   " +
            f"{roi_new:>4}" + GRAY + " |" + Style.RESET_ALL + "  " +
            f"{uid_total:>5}" + "   " +
            f"{uid_dup:>4}" + "   " +
            f"{uid_new:>4}"
        )
        print(row)

    print(colorize_line(underline))
    totals_row = (
        " TOTALS:".rjust(7) + "  " +
        f"{tot_rec:>5}" + "   " +
        f"{tot_svr:>5}" + "   " +
        f"{tot_mvr:>4}" + "   " +
        f"{tot_amb:>3}" + GRAY + "|" + Style.RESET_ALL + "  " +
        f"{tot_roi_tot:>5}" + "   " +
        f"{tot_roi_dup:>3}" + "   " +
        f"{tot_roi_new:>4}" + GRAY + " |" + Style.RESET_ALL + "  " +
        f"{tot_uid_tot:>5}" + "   " +
        f"{tot_uid_dup:>4}" + "   " +
        f"{tot_uid_new:>4}"
    )
    print(totals_row)
    print(colorize_line(underline))
    print()

    # ── 2) Detailed statistics ────────────────────────────────────────
    print("@Writing detailed statistics …")
    stats_text, _ = report.generate_statistics(total_records)
    stats_path = os.path.join(report_dir, "initial_statistics.txt")
    report.write_report(stats_text, stats_path)
    print(f"  Detailed stats → {stats_path}")

    # ── 3) Splitting report ──────────────────────────────────────────
    print("@Writing splitting report …")
    write_splitting_report(splitting_events, report_dir)

    # ── 4) Custom summary ────────────────────────────────────────────
    print("@Writing custom summary …")
    write_custom_summary(total_records, counts, report_dir)

    # ── 5) Build rois_by_uid ─────────────────────────────────────────
    rois_by_uid = defaultdict(set)
    for r in total_records:
        rois_by_uid[r["uid"]].add(r["roi"])

    # ── 6) Fetch & validate sequences + extract windows ──────────────
    lookup = (
        "local FASTA"
        if _distill._UNIPROT_LOCAL_DB is not None
        else "UniProt REST"
    )
    print(f"@Fetching & validating sequences (primary lookup: {lookup}) …")

    # 6a) Parallel fetch of full sequences (no ROI validation yet)
    uid_list = list(rois_by_uid.keys())

    uniprot_fail_path = os.path.join(report_dir, "unavailable_uniprotids.txt")
    pdb_fail_path = os.path.join(report_dir, "unavailable_pdbids.txt")
    unacceptable_path = os.path.join(report_dir, "unacceptable_sequences.fasta")
    validation_fail_path = os.path.join(report_dir, "validation_failures.txt")

    uniprot_fail_file = open(uniprot_fail_path, "w")
    pdb_fail_file = open(pdb_fail_path, "w")
    unacceptable_file = open(unacceptable_path, "w")
    validation_file = open(validation_fail_path, "w")

    valid_sequences = {}
    with Pool(processes=threads) as pool:
        for uid, full_seq, err in tqdm(
            pool.imap_unordered(fetch_sequence_only, uid_list),
            total=len(uid_list),
            desc="  Fetching sequences",
            ncols=80
        ):
            if err:
                validation_file.write(f"{uid}\t{err}\n")
                low = err.lower()
                if is_pdb_uid(uid):
                    if "download" in low or "failed" in low:
                        pdb_fail_file.write(f"{uid}\n")
                    elif "sequence contains" in low and "?" in low:
                        unacceptable_file.write(f">{uid}\n")
                    else:
                        pdb_fail_file.write(f"{uid}\n")
                else:
                    if low.startswith("http") or "request" in low:
                        uniprot_fail_file.write(f"{uid}\n")
                    elif "sequence contains" in low and "x" in low:
                        unacceptable_file.write(f">{uid}\n")
                    else:
                        uniprot_fail_file.write(f"{uid}\n")
                continue
            valid_sequences[uid] = full_seq

    uniprot_fail_file.close()
    pdb_fail_file.close()
    unacceptable_file.close()
    validation_file.close()

    # ── Cache sequences for ROI-validation phase ──────────────────────
    # This makes downstream fetch() pull from memory, not the network.
    _distill._UNIPROT_LOCAL_DB = valid_sequences

    # 6b) Per-record ROI validation, dropping (mv)/(am) individually
    invalid_path = os.path.join(report_dir, "invalid_records.txt")
    invalid_fh = open(invalid_path, "w")

    def write_invalid(rec):
        invalid_fh.write(json.dumps(rec) + "\n")

    per_uid = defaultdict(list)
    for r in total_records:
        per_uid[r["uid"]].append(r)

    accepted_records = []

    # Stage 1: strict per-UID validation using worker
    with Pool(processes=threads) as pool:
        for uid, full_seq, roi_map, err in tqdm(
            pool.imap_unordered(
                fetch_sequence_worker, list(rois_by_uid.items())
            ),
            total=len(rois_by_uid),
            desc="  Fetching & validating (strict per-UID)",
            ncols=80
        ):
            if err:
                # any-ROI error → drop entire UID here
                continue
            # All ROIs mapped to Cys → keep those records as-is
            for r in per_uid[uid]:
                mapped_idx = roi_map[r["roi"]]
                window_seq = compute_window(full_seq, mapped_idx, window)
                new_r = dict(r)
                new_r["sequence"] = full_seq
                new_r["mapped_roi"] = mapped_idx
                new_r["window_seq"] = window_seq
                accepted_records.append(new_r)

    # Stage 2: rescue (mv)/(am) or, if nostrict, normals too
    dropped_uids = set(rois_by_uid) - {r["uid"] for r in accepted_records}

    for uid in tqdm(
        dropped_uids,
        desc="  Rescuing (mv)/(am) if nostrict",
        ncols=80
    ):
        # If sequence fetch failed, skip entirely
        if uid not in valid_sequences:
            for r in per_uid[uid]:
                write_invalid(r)
            continue

        full_seq = valid_sequences[uid]
        recs = per_uid[uid]
        saw_normal_mismatch = False
        rescued = []
        for r in recs:
            roi = r["roi"]
            note = r.get("note", "")
            is_expanded = note.endswith("(mv)") or note.endswith("(am)")

            # Check boundary and Cys
            if roi < 1 or roi > len(full_seq) or full_seq[roi - 1] != "C":
                write_invalid(r)
                if not is_expanded:
                    # “Normal” mismatch
                    if not nostrict:
                        saw_normal_mismatch = True
                        break
                    else:
                        # nostrict=True → drop only this record
                        continue
                else:
                    # Expanded record → drop only this record
                    continue

            # Valid (UID, ROI)
            window_seq = compute_window(full_seq, roi, window)
            new_r = dict(r)
            new_r["sequence"] = full_seq
            new_r["mapped_roi"] = roi
            new_r["window_seq"] = window_seq
            rescued.append(new_r)

        if saw_normal_mismatch and not nostrict:
            # In strict mode, any normal mismatch drops entire UID
            continue

        if rescued:
            accepted_records.extend(rescued)
        else:
            continue

    invalid_fh.close()

    # ── 7) Count before writing FASTA ─────────────────────────────────
    total_post_split = len(total_records)
    accepted_pairs = len({(r["uid"], r["roi"]) for r in accepted_records})

    # ── 8) Build per-UID mapping from accepted_records ───────────────
    uid_map = {}  # uid → {"sequence": seq, "roi_map": {orig_roi: mapped_roi}}
    for r in accepted_records:
        uid = r["uid"]
        mapped = r["mapped_roi"]
        seq = r["sequence"]
        if uid not in uid_map:
            uid_map[uid] = {"sequence": seq, "roi_map": {}}
        uid_map[uid]["roi_map"][r["roi"]] = mapped

    # ── 9) Write uid_sequences.fasta with a tqdm bar ─────────────────
    seqs_fasta = os.path.join(report_dir, "uid_sequences.fasta")
    try:
        with open(seqs_fasta, "w") as fasta_file:
            for uid, info in tqdm(
                uid_map.items(),
                desc="Writing uid_sequences.fasta",
                total=len(uid_map),
                ncols=80
            ):
                full_seq = info["sequence"]
                roi_map = info["roi_map"]  # orig_roi → mapped_idx

                mapping_str = ",".join(
                    f"{o}->{m}" for o, m in sorted(roi_map.items())
                )
                roi_str = ",".join(
                    str(m) for _, m in sorted(roi_map.items())
                )
                roi_type = "LC3D" if "-" in uid else "ABPP"

                header = f">{uid} {roi_type}={roi_str} MAPS={mapping_str}"
                fasta_file.write(header + "\n")
                fasta_file.write(full_seq + "\n")
        _info(f"uid_sequences.fasta written: {seqs_fasta}", C.CYAN)
    except Exception as e:
        _info(
            f"Warning: could not write '{seqs_fasta}': {e}",
            C.YELLOW
        )

    # ── 10) Final return ─────────────────────────────────────────────
    return accepted_records, total_post_split, accepted_pairs

