#!/usr/bin/env python3
# distill.py
# Accelerated, memory‐ and compute‐efficient cysteine‐grouping for ABPP
# denoising. Supports “pure PDB” mode when --usepdbs: fetches PDB, parses
# via pdbdr.

import os
import json
import urllib.request
import requests
import torch
import numpy as np
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from tqdm import tqdm
from collections import defaultdict
import math

from .parse import (
    parse_bool,
    parse_threshold_arg,
    parse_cb_arg,
    load_data,
    parse_arguments,
    setup_thresholds,
)

from pdbdr.esm_filling import pdb_to_tensor  # your pdbdr-based parser

# Globals set in main()
DEBUG = False
USE_PDBS = False  # must be set by your CLI: distill.USE_PDBS = args.usepdbs


# -----------------------------------------------------------------------------
# Optional in-memory UniProt FASTA. If set, all fetch_uniprot_sequence() and
# _fetch_sequence() calls will pull from this dict instead of making HTTP
# requests. This avoids network latency and flakiness.
_UNIPROT_LOCAL_DB = None


def set_uniprotdb(fasta_path: str):
    """
    Read the given FASTA into memory (UID → sequence). Simple and fast way
    to avoid hitting UniProt servers repeatedly.
    """
    global _UNIPROT_LOCAL_DB
    _UNIPROT_LOCAL_DB = {}
    with open(fasta_path) as fh:
        uid = None
        seq_lines = []
        for line in fh:
            if line.startswith(">"):
                if uid:
                    _UNIPROT_LOCAL_DB[uid] = "".join(seq_lines)
                uid = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if uid:
            _UNIPROT_LOCAL_DB[uid] = "".join(seq_lines)
    print(
        f"[uniprot] Loaded local UniProt DB from {fasta_path}: "
        f"{len(_UNIPROT_LOCAL_DB)} sequences available"
    )


def get_pdb_file(pdb_id: str, cache_dir: str = "distiller_pdbs") -> str:
    """
    Download a PDB file from RCSB (fallback PDBe) and cache it locally so
    we do not re-download it next time.
    """
    pdb_id_uc = pdb_id.upper()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id_uc}.pdb")
    if os.path.isfile(out_path):
        return out_path

    # Try RCSB first (typical primary source).
    url1 = f"https://files.rcsb.org/download/{pdb_id_uc}.pdb"
    try:
        r = requests.get(url1, timeout=10)
        if r.status_code == 200 and r.text.startswith("HEADER"):
            with open(out_path, "w") as f:
                f.write(r.text)
            return out_path
    except Exception:
        pass

    # Fallback to PDBe if RCSB fails (redundant mirror).
    url2 = (
        f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id.lower()}.pdb"
    )
    try:
        r2 = requests.get(url2, timeout=10)
        if r2.status_code == 200 and r2.text.startswith("HEADER"):
            with open(out_path, "w") as f:
                f.write(r2.text)
            return out_path
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not download PDB {pdb_id_uc} from RCSB or PDBe."
    )


def _load_embedding_file(args):
    """
    Internal loader. Reads a single .npy or .pt embedding file, extracts
    only the positions we actually need, and returns a mapping.
    """
    path, repr_layers, needed = args
    name, ext = os.path.splitext(os.path.basename(path))
    uids = name.split(",")
    positions = set()
    for u in uids:
        if u in needed:
            positions |= needed[u]
    if not positions:
        return None
    try:
        if ext == ".npy":
            arr = np.load(path)
            emb_map = {p: arr[p - 1] for p in positions if 1 <= p <= arr.shape[0]}
            return uids, emb_map

        data = torch.load(path, map_location="cpu")
        layers = []
        if isinstance(data, dict) and "representations" in data:
            reps = data["representations"]
            for L in repr_layers:
                if L not in reps:
                    continue
                lm = reps[L]
                if isinstance(lm, torch.Tensor):
                    layers.append(lm.cpu().to(torch.float32))
                elif isinstance(lm, dict):
                    max_pos = max(int(p) for p in lm.keys())
                    D = next(iter(lm.values())).size(-1)
                    T = torch.zeros((max_pos, D), dtype=torch.float32)
                    for p, v in lm.items():
                        T[int(p) - 1] = v.cpu().to(torch.float32)
                    layers.append(T)
        elif isinstance(data, torch.Tensor):
            layers.append(data.cpu().to(torch.float32))

        if not layers:
            return None

        esm_t = torch.cat(layers, dim=1)
        emb_map = {
            p: esm_t[p - 1].numpy()
            for p in positions
            if 1 <= p <= esm_t.size(0)
        }
        return uids, emb_map

    except Exception as e:
        if DEBUG:
            print(f"[load_embeddings] error on {name}: {e}")
        return None


def load_embeddings(embs_dir, repr_layers, needed, max_workers=None):
    """
    Load many embedding files and keep only the residues we need for this
    run. Uses multiple processes for speed on big directories.
    """
    files = sorted(
        fn for fn in os.listdir(embs_dir)
        if fn.endswith(".npy") or fn.endswith(".pt")
    )
    args_list = [
        (os.path.join(embs_dir, fn), repr_layers, needed) for fn in files
    ]
    embs = {}
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers == 1:
        for arg in tqdm(args_list, desc="Loading embeddings", ncols=80):
            out = _load_embedding_file(arg)
            if not out:
                continue
            uids, emb_map = out
            for u in uids:
                if u in needed:
                    embs[u] = emb_map
        return embs

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_load_embedding_file, arg): arg for arg in args_list}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading embeddings",
            ncols=80,
        ):
            out = fut.result()
            if not out:
                continue
            uids, emb_map = out
            for u in uids:
                if u in needed:
                    embs[u] = emb_map
    return embs


def filter_proteins_by_cbseq(records, cbseq_rule):
    """
    Keep only proteins that pass a CB-sequence rule on at least one TRUE
    record. Quick way to focus on proteins with sufficient TRUE support.
    """

    def _cb_rule_pass(count, rule):
        op, thresh = rule
        if op == "==":
            return count == thresh
        if op == ">":
            return count > thresh
        if op == ">=":
            return count >= thresh
        if op == "<":
            return count < thresh
        if op == "<=":
            return count <= thresh
        return False

    per_uid = defaultdict(list)
    for r in records:
        per_uid[r["uid"]].append(r)
    keep = set()
    for uid, rows in per_uid.items():
        for r in rows:
            if r["exp_bin"] and _cb_rule_pass(r.get("cb_true_src", 0), cbseq_rule):
                keep.add(uid)
                break
    return [r for r in records if r["uid"] in keep]


def fetch_uniprot_sequence(uniprot_id):
    """
    Get a UniProt sequence as plain text. Uses in-memory DB first; if it is
    missing we fetch online. Returns just the sequence string.
    """
    # 1) try local first
    if _UNIPROT_LOCAL_DB is not None:
        seq = _UNIPROT_LOCAL_DB.get(uniprot_id)
        if seq is not None:
            return seq
        # 2) local miss → bubble up above any tqdm bars
        tqdm.write(
            f"[uniprot miss] {uniprot_id} not in local DB, fetching online…"
        )

    # 3) fallback to network
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            fasta = resp.read().decode()
    except Exception as e:
        if DEBUG:
            print(f"Error fetching sequence for {uniprot_id}: {e}")
        return None
    return "".join(line for line in fasta.splitlines() if not line.startswith(">"))


def map_pdb_to_uniprot(uid, pdb_roi):
    """
    Map a PDB residue position to UniProt numbering using PDBe mappings.
    Returns (uniprot_id, uniprot_pos) or (None, None) if not mapped.
    """
    # legacy branch (unused when USE_PDBS=True)
    parts = uid.split("-")
    if len(parts) != 2:
        return None, None
    pdb_id, chain_id = parts[0].upper(), parts[1].upper()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        if DEBUG:
            print(f"Error fetching mapping for PDB {pdb_id}: {e}")
        return None, None
    pdb_data = data.get(pdb_id.lower(), {})

    def extract_residue(val):
        return val.get("residue_number") if isinstance(val, dict) else val

    if "UniProt" in pdb_data:
        for unp, info in pdb_data["UniProt"].items():
            for m in info.get("mappings", []):
                chain = m.get("chain_id") or m.get("chain")
                if chain and chain.upper() == chain_id:
                    start = extract_residue(
                        m.get("start_residue") or m.get("start")
                    )
                    end = extract_residue(m.get("end_residue") or m.get("end"))
                    up0 = m.get("unp_start") or m.get("uniprot_start")
                    if None not in (start, end, up0):
                        start, end, up0 = int(start), int(end), int(up0)
                        if start <= pdb_roi <= end:
                            return unp, up0 + (pdb_roi - start)
    for m in pdb_data.get("mappings", []):
        chain = m.get("chain_id") or m.get("chain")
        if chain and chain.upper() == chain_id:
            start = extract_residue(m.get("start_residue") or m.get("start"))
            end = extract_residue(m.get("end_residue") or m.get("end"))
            unp = m.get("unp_id") or m.get("uniprot_id")
            up0 = m.get("unp_start") or m.get("uniprot_start")
            if None not in (start, end, unp, up0):
                start, end, up0 = int(start), int(end), int(up0)
                if start <= pdb_roi <= end:
                    return unp, up0 + (pdb_roi - start)
    return None, None


def process_pdb_record(rec):
    """
    Map a single PDB-mode record. In pure PDB mode we keep UID/ROI as-is.
    Otherwise we resolve to UniProt and validate cysteine identity.
    """
    if USE_PDBS:
        rec["mapped"] = True
        rec["pdb_mapping_valid"] = True
        rec["pdb_mapped_uid"] = rec["uid"]
        rec["pdb_mapped_roi"] = rec["roi"]
        return rec, "success", rec["uid"], rec["roi"]

    new_unp, new_roi = map_pdb_to_uniprot(rec["uid"], rec["roi"])
    if new_unp is not None and new_roi is not None:
        seq = fetch_uniprot_sequence(new_unp)
        if not seq or new_roi > len(seq) or seq[new_roi - 1] != "C":
            rec["mapped"] = False
            rec["pdb_mapping_valid"] = False
            return rec, "mismatch"
        rec["mapped"] = True
        rec["pdb_mapping_valid"] = True
        return rec, "success", new_unp, new_roi

    rec["mapped"] = False
    rec["pdb_mapping_valid"] = False
    return rec, "fail"


def process_pdb_mappings(total_records, args):
    """
    Process all candidate PDB records in parallel. Produces a filtered
    record list and a human-readable summary report string.
    """
    pdb_recs = [
        r for r in total_records if r["R"] == 999 and r["exp_thr"] == 0 and "-"
        in r["uid"]
    ]
    tcount = len(pdb_recs)
    succ, faild, mm = 0, [], []

    if pdb_recs:
        max_workers = min(64, len(pdb_recs))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futs = {exe.submit(process_pdb_record, r): r for r in pdb_recs}
            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc="Mapping PDB entries",
                ncols=80,
                ascii=True,
            ):
                rec, status, *extra = fut.result()
                if status == "success":
                    succ += 1
                    new_uid, new_roi = extra
                    if args.usepdbs:
                        rec["pdb_mapped_uid"] = new_uid
                        rec["pdb_mapped_roi"] = new_roi
                    else:
                        old = rec["uid"]
                        rec["uid"], rec["roi"], rec["note"] = (
                            new_uid,
                            new_roi,
                            old,
                        )
                elif status == "mismatch":
                    mm.append(rec["uid"])
                else:
                    faild.append(rec["uid"])
    else:
        print("No PDB entries found.")

    filtered = []
    for r in total_records:
        if r["R"] == 999 and r["exp_thr"] == 0:
            if not r.get("mapped"):
                continue
            if not USE_PDBS and not r.get("pdb_mapping_valid", True):
                continue
        filtered.append(r)

    report = [
        "PDB Mapping Report",
        "==================",
        f"Total PDB entries: {tcount}",
        f"Mapped successfully: {succ}",
        f"Failed mapping: {len(faild)}",
        f"Cys mismatch: {len(mm)}",
    ]
    if mm:
        report.append("Mismatched: " + ", ".join(sorted(set(mm))))
    return filtered, "\n".join(report)


def filter_conflicted_records(records):
    """
    Remove records with conflicting positive/negative calls at the same
    (uid, roi) unless they are explicitly marked ambiguous/multi_value.
    """
    label_map = defaultdict(set)
    for r in records:
        if r.get("ambiguous"):
            continue
        for uid in str(r["uid"]).split(";"):
            for roi in str(r["roi"]).split(";"):
                label_map[(uid, roi)].add(r["exp_bin"])
    conflicted = {k for k, v in label_map.items() if len(v) > 1}
    out = []
    for r in records:
        if r.get("ambiguous") or r.get("multi_value"):
            out.append(r)
        else:
            key = (str(r["uid"]), str(r["roi"]))
            if key not in conflicted:
                out.append(r)
    return out


def filter_ambiguous_records(records):
    """
    Keep only clean records (drop ambiguous or multi_value ones).
    """
    return [
        r for r in records if not (r.get("ambiguous") or r.get("multi_value"))
    ]


def write_distilled_dataset(records, filename="distilled_dataset.dat"):
    """
    Write a tab-delimited file of deduplicated (uid, roi) pairs with key
    fields. This is a compact summary used downstream.
    """
    seen = set()
    with open(filename, "w") as f:
        f.write(
            "\t".join(
                [
                    "uid",
                    "roi",
                    "R",
                    "exp_thr",
                    "exp_bin",
                    "source",
                    "note",
                    "label",
                ]
            )
            + "\n"
        )
        for r in records:
            key = (r["uid"], r["roi"])
            if key in seen:
                continue
            seen.add(key)
            lbl = "1" if r["exp_bin"] else "0"
            f.write(
                "\t".join(
                    [
                        r["uid"],
                        f"{r['roi']:4d}",
                        f"{r['R']:5.2f}",
                        str(r["exp_thr"]),
                        str(r["exp_bin"]),
                        r["source"],
                        f"{r['note']:15s}",
                        lbl,
                    ]
                )
                + "\n"
            )
    print(
        f"Deduplication: wrote {len(seen)} unique records (skipped "
        f"{len(records)-len(seen)})"
    )


def write_report(report_text, filename):
    """
    Save a plain-text report to disk. Simple helper for pipeline logs.
    """
    with open(filename, "w") as f:
        f.write(report_text + "\n")


def apply_filters(
    final_distilled,
    args,
    pos_min_comp,
    pos_min_func,
    pos_max_comp,
    pos_max_func,
    neg_max_comp,
    neg_max_func,
    neg_min_comp,
    neg_min_func,
    comp_ops,
):
    """
    Apply numeric and source-based filters to the distilled records in a
    deterministic order. Each step prints how many survive.
    """
    if args.rmax is not None:
        clamp = args.rmax + 0.01
        for r in final_distilled:
            if r["R"] > args.rmax:
                r["R"] = clamp
        print(f"Clamped R>={args.rmax} to {clamp}.")
    if pos_min_func or pos_max_func or neg_max_func or neg_min_func:
        tmp = []
        for r in final_distilled:
            if r["R"] == 999 and r["exp_thr"] == 0:
                tmp.append(r)
                continue
            ok = True
            if r["exp_bin"]:
                if pos_min_func:
                    th = pos_min_func(r["exp_thr"], r["R"])
                    if not comp_ops[pos_min_comp](r["R"], th):
                        ok = False
                if ok and pos_max_func:
                    th = pos_max_func(r["exp_thr"], r["R"])
                    if not comp_ops[pos_max_comp](r["R"], th):
                        ok = False
            else:
                if neg_max_func:
                    th = neg_max_func(r["exp_thr"], r["R"])
                    if not comp_ops[neg_max_comp](r["R"], th):
                        ok = False
                if ok and neg_min_func:
                    th = neg_min_func(r["exp_thr"], r["R"])
                    if not comp_ops[neg_min_comp](r["R"], th):
                        ok = False
            if ok:
                tmp.append(r)
        final_distilled = tmp
        print(f"After R threshold filters, {len(final_distilled)} records.")
    tmp = []
    for r in final_distilled:
        mw = r.get("mw")
        if mw is not None and (mw < args.minMW or mw > args.maxMW):
            continue
        tmp.append(r)
    final_distilled = tmp
    print(f" After MW filtering, {len(final_distilled)} records.")
    if not args.allownegonly:
        pos_uids = {x["uid"] for x in final_distilled if x["exp_bin"]}
        final_distilled = [
            x for x in final_distilled if x["exp_bin"] or x["uid"] in pos_uids
        ]
        print(f" Removed no-positive negatives, {len(final_distilled)} left.")
    if args.exclude:
        if args.exclude.endswith(".dat") and os.path.isfile(args.exclude):
            with open(args.exclude) as f:
                lines = f.readlines()
            hdr = lines[0].split()
            uid_idx = hdr.index("uid") if "uid" in hdr else 0
            ex = {ln.split()[uid_idx] for ln in lines[1:] if ln.strip()}
        else:
            tokens = [t.strip() for t in args.exclude.split(",") if t.strip()]
            ex = {
                r["uid"]
                for r in final_distilled
                if any(tok in r["source"] for tok in tokens)
            }
        final_distilled = [r for r in final_distilled if r["uid"] not in ex]
        print(f" After excludes, {len(final_distilled)} records.")
    if args.prioritize:
        prefs = {p.strip() for p in args.prioritize.split(",") if p.strip()}
        final_distilled = [r for r in final_distilled if r["source"] in prefs]
        print(f" After prioritize filter, {len(final_distilled)} records.")
    return final_distilled


def load_fasta(seqs_fasta):
    """
    Load a FASTA of UID → sequence. If a header has multiple comma-separated
    UIDs, each alias maps to the same sequence.
    """
    seqs_raw, uid, lines = {}, None, []
    with open(seqs_fasta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if uid:
                    seqs_raw[uid] = "".join(lines)
                uid = line[1:].split()[0]
                lines = []
            else:
                lines.append(line)
        if uid:
            seqs_raw[uid] = "".join(lines)

    seqs = {}
    for raw_uid, seq in seqs_raw.items():
        for sub in raw_uid.split(","):
            seqs[sub] = seq
    return seqs


def compute_window(seq, pos, window):
    """
    Extract a centered window around a 1-based residue position. Simple
    bounds handling avoids index errors at sequence ends.
    """
    half = window // 2
    start = max(pos - half - 1, 0)
    end = pos + half
    return seq[start:end]


def seq_identity(w1, w2):
    """
    Percent identity between two windows. Shorter window length sets the
    comparison span. If one is empty, identity is zero.
    """
    L = min(len(w1), len(w2))
    if L == 0:
        return 0.0
    matches = sum(1 for a, b in zip(w1[:L], w2[:L]) if a == b)
    return matches / L * 100.0


def comp_similarity(v1, v2):
    """
    Composite embedding similarity combining cosine and inverse distances.
    Gives a bounded score in (0, 1+] where higher means more similar.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    vi, vj = v1 / n1, v2 / n2
    cos_ij = float(np.dot(vi, vj))
    diff = v1 - v2
    d2 = float(np.linalg.norm(diff, ord=2))
    d1 = float(np.linalg.norm(diff, ord=1))
    s2 = 1.0 / (1.0 + d2)
    s1 = 1.0 / (1.0 + d1)
    return (cos_ij + s2 + s1) / 3.0


def _fetch_sequence(uid, cache_dir):
    """
    For uid:
      - If of form PDBID-CHAIN, download & parse via pdbdr.
      - Else treat as UniProt and fetch via UniProt REST.
    Returns (uid, full_seq, idx_map) where idx_map maps (resSeq,"")→position.
    """
    if "-" in uid:
        # PDB branch
        pdb_id, chain = uid.split("-", 1)
        pdb_file = get_pdb_file(pdb_id, cache_dir)
        _, full_seq, _, seq_pos = pdb_to_tensor(pdb_file, chain)
        inv = {res_id: idx for idx, res_id in seq_pos.items()}
        return uid, full_seq, inv
    else:
        # UniProt branch: either from local DB or from web
        if _UNIPROT_LOCAL_DB is not None:
            seq = _UNIPROT_LOCAL_DB.get(uid)
        else:
            seq = fetch_uniprot_sequence(uid)

        if seq is None:
            raise FileNotFoundError(
                f"Failed to fetch UniProt sequence for {uid}"
            )
        # Map each residue number → index (simple identity mapping).
        inv = {(i, ""): i for i in range(1, len(seq) + 1)}
        return uid, seq, inv


# ─────────────────────────────────────────────────────────────────────────────
# In aipp/distill.py: replace the existing `build_cys_groups` with this
# version. Kept verbatim content below, just wrapped and commented for
# readability.
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import torch
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# (Keep your existing imports: pdbdr, requests, etc.)

def build_cys_groups(
    records,
    seqs_file,
    embs_dir,
    window,
    id_thr,
    comp_thr,
    repr_layers,
    max_workers=None,
    block_size=512,
):
    """
    2D‐blocked, GPU‐accelerated grouping of cysteines—but now with clear
    progress bars on FASTA loading and node‐building.

    records: list of dicts (uid, roi, R, exp_thr, exp_bin, source, note,
             mw, og_uid, sequence, mapped_roi, window_seq)
    seqs_file: path to 'uid_sequences.fasta'
    embs_dir: directory of .npy or .pt embeddings
    window: odd integer window size
    id_thr: sequence identity threshold (percent)
    comp_thr: composite similarity threshold
    repr_layers: list of integers (ESM layers)
    max_workers: for parallel embedding loading
    block_size: GPU block size (default 512; raise if GPU has more RAM)
    """
    # ── 1) Determine needed embeddings (exact same as before) ────────────────
    needed = defaultdict(set)
    for r in records:
        base = r["uid"].split("-", 1)[0]
        needed[base].add(r["roi"])

    # ── 2) Load embeddings (monkey‐patched in some pipelines) ────────────────
    embs = load_embeddings(embs_dir, repr_layers, needed, max_workers)

    # ── 3) Prepare sequences ─────────────────────────────────────────────────
    # We do NOT do PDB branch here (USE_PDBS=False in our entry script).
    print("Loading FASTA from disk:", seqs_file)
    seqs = load_fasta(seqs_file)  # pure‐Python I/O

    # ── 4) Build nodes (window + embedding vector) ───────────────────────────
    nodes = []
    print("Building nodes (extracting subsequences + grabbing embeddings)…")
    for r in tqdm(records, desc="  Building nodes", ncols=80):
        base = r["uid"].split("-", 1)[0]
        emb_map = embs.get(base)
        if emb_map is None:
            continue

        # Fetch the full sequence and compute the centered window.
        seq = seqs.get(base)
        if seq is None:
            continue

        start = max(r["mapped_roi"] - (window // 2) - 1, 0)
        end = r["mapped_roi"] + (window // 2)
        w = seq[start:end]

        # Get the embedding vector at the ROI.
        vec = emb_map.get(r["roi"])
        if vec is not None and len(w) > 0:
            nodes.append({"rec": r, "window": w, "vec": vec})

    M = len(nodes)
    if M == 0:
        return []

    print(f" → {M} nodes total")

    # ── 5) Build W_int and X tensors on device ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Assembling W_int and X tensors…")

    aa2int = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    W_int = torch.full((M, window), -1, dtype=torch.int8, device=device)
    for i, nd in enumerate(tqdm(nodes, desc="  Encoding windows", ncols=80)):
        arr = [aa2int.get(c, -1) for c in nd["window"]]
        if len(arr) < window:
            arr += [-1] * (window - len(arr))
        W_int[i] = torch.tensor(arr, dtype=torch.int8, device=device)

    X = torch.stack(
        [torch.from_numpy(nd["vec"]) for nd in nodes], dim=0
    ).to(device)
    Xn = X / X.norm(dim=1, keepdim=True)
    print("  Tensors ready on", device)

    # ── 6) Compute edges with 2D blocks (GPU) ────────────────────────────────
    bs = min(block_size, M)
    block_pairs = [
        (i0, min(i0 + bs, M), j0, min(j0 + bs, M))
        for i0 in range(0, M, bs)
        for j0 in range(i0, M, bs)
    ]

    edges = []
    print("Computing edges (this is GPU‐accelerated)…")
    for (i0, i1, j0, j1) in tqdm(
        block_pairs,
        desc="  Computing edges",
        total=len(block_pairs),
        ncols=80,
        leave=True,
    ):
        Xi, Xni, Wi = X[i0:i1], Xn[i0:i1], W_int[i0:i1]
        Xj, Xnj, Wj = X[j0:j1], Xn[j0:j1], W_int[j0:j1]

        eq = (Wi.unsqueeze(1) == Wj.unsqueeze(0)).sum(dim=2).float()
        id_mat = eq / window * 100.0

        cos = Xni @ Xnj.T
        d2 = torch.cdist(Xi, Xj, p=2)
        s2 = 1.0 / (1.0 + d2)
        d1 = torch.cdist(Xi, Xj, p=1)
        s1 = 1.0 / (1.0 + d1)
        comp = (cos + s2 + s1) / 3.0

        mask = (id_mat >= id_thr) | (comp >= comp_thr)
        idxs = mask.nonzero(as_tuple=False)
        for bi, bj in idxs:
            i = i0 + int(bi.item())
            j = j0 + int(bj.item())
            if j > i:
                edges.append((i, j))

        # Free memory for next block
        del Xi, Xni, Wi, Xj, Xnj, Wj, eq, id_mat, cos, d2, s2, d1, s1, comp
        del mask, idxs
        torch.cuda.empty_cache()

    # ── 7) Union‐Find to assemble clusters ───────────────────────────────────
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    clusters = defaultdict(list)
    for idx, nd in enumerate(nodes):
        gid = find(idx)
        clusters[gid].append((idx, nd))

    # ── 8) Build the final group_list with reps & member stats ───────────────
    group_list = []
    print("Forming cluster summaries…")
    for gid, members in tqdm(
        clusters.items(), desc="  Forming clusters", ncols=80
    ):
        sims = {}
        for idx, nd in members:
            total_sim = 0.0
            count = 0
            for jdx, nd2 in members:
                if idx == jdx:
                    continue
                total_sim += comp_similarity(nd["vec"], nd2["vec"])
                count += 1
            sims[idx] = total_sim / count if count else 0.0

        rep_idx = max(sims, key=sims.get)
        rep_rec = nodes[rep_idx]["rec"]
        rep = {"uid": rep_rec["uid"], "roi": rep_rec["roi"]}

        items = []
        rep_w, rep_v = nodes[rep_idx]["window"], nodes[rep_idx]["vec"]
        for idx, nd in members:
            rec = nd["rec"]
            items.append(
                {
                    "uid": rec["uid"],
                    "roi": rec["roi"],
                    "identity": seq_identity(rep_w, nd["window"]),
                    "comp": comp_similarity(rep_v, nd["vec"]),
                    "label": "TRUE" if rec["exp_bin"] else "FALSE",
                    "source": rec["source"],
                    "note": rec.get("note", ""),
                    "R": rec["R"],
                }
            )

        group_list.append(
            {
                "group_id": gid,
                "representative": rep,
                "members": items,
            }
        )

    return group_list


# End of patched build_cys_groups


def write_cys_groups_report(groups, filename):
    """
    Write a human-readable report for each cluster: representative and all
    member entries with their stats. Handy for manual inspection.
    """
    lines = ["Cysteine grouping report", "========================="]
    for g in groups:
        rep = g["representative"]
        lines.append(f"Group {g['group_id']}: repr={rep['uid']}:{rep['roi']}")
        for m in g["members"]:
            lines.append(
                f"  {m['uid']}:{m['roi']} "
                f"id={m['identity']:.1f}%, comp={m['comp']:.3f}, "
                f"label={m['label']}, source={m['source']}, "
                f"note={m['note']}, R={m['R']:.2f}"
            )
        lines.append("")
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def group_unambiguous_records(records):
    """
    Group records by exact (uid, roi) pairs, except in PDB mode where the
    mapped UID/ROI are used. This is a strict grouping step.
    """
    groups = defaultdict(list)
    for r in records:
        if USE_PDBS and r.get("mapped") and r.get("pdb_mapping_valid"):
            key = (r["pdb_mapped_uid"], r["pdb_mapped_roi"])
        else:
            key = (r["uid"], r["roi"])
        groups[key].append(r)
    return groups


def _summarize_single_cluster(args):
    """
    Worker function for one cluster. Unpacked from (gid, member_list, nodes,
    id_thr, comp_thr). Returns a dict in the same format that
    group_precomputed_nodes expects for a single group.
    """
    gid, members, nodes = args
    # `members` is a list of (idx, nd) where nd["vec"] is a numpy array,
    # nd["window"] is a string. `nodes` is the full list of node-dicts.

    # 1) Compute “sims” (avg similarity to others) for each member:
    sims = {}
    for idx, nd in members:
        tot = 0.0
        cnt = 0
        v_i = nd["vec"]  # NumPy array
        for jdx, nd2 in members:
            if idx == jdx:
                continue
            v_j = nd2["vec"]
            tot += comp_similarity(v_i, v_j)
            cnt += 1
        sims[idx] = (tot / cnt) if (cnt > 0) else 0.0

    # 2) Representative = member with largest avg similarity:
    rep_idx = max(sims, key=sims.get)
    rep_rec = nodes[rep_idx]["rec"]
    rep = {"uid": rep_rec["uid"], "roi": rep_rec["roi"]}

    # 3) Build “items” for this cluster (expanded per record):
    items = []
    rep_w = nodes[rep_idx]["window"]
    rep_v = nodes[rep_idx]["vec"]
    for idx, nd in members:
        rec = nd["rec"]
        items.append(
            {
                "uid": rec["uid"],
                "roi": rec["roi"],
                "identity": seq_identity(rep_w, nd["window"]),
                "comp": comp_similarity(rep_v, nd["vec"]),
                "label": "TRUE" if rec["exp_bin"] else "FALSE",
                "source": rec["source"],
                "note": rec.get("note", ""),
                "R": rec["R"],
            }
        )

    return {"group_id": gid, "representative": rep, "members": items}


def _encode_window_to_int_pair(args):
    """
    Helper for multiprocessing Pool. args = (wstr, window). Returns a
    NumPy array of shape (window,) dtype=np.int8 with integer codes for
    each amino acid, or -1 if unknown. Keeps memory compact.
    """
    wstr, window = args
    aa2int = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    arr = [aa2int.get(c, -1) for c in wstr]
    if len(arr) < window:
        arr += [-1] * (window - len(arr))
    # If wstr is longer than `window` (unlikely here), we simply truncated in
    # load_dataset, so arr should never exceed length `window`.
    return np.array(arr, dtype=np.int8)


def group_precomputed_nodes(
    nodes,
    window,
    id_thr,
    comp_thr,
    n_workers=None,  # number of processes for cluster summaries
    block_size=512,
):
    """
    Sister function to build_cys_groups(), but with parallel cluster
    summaries. Use when nodes are already built in memory.
    """
    M = len(nodes)
    if M == 0:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    # ── 1) Assemble W_int & X tensors ────────────────────────────────────────
    print("Assembling W_int and X tensors (GPU) …", flush=True)
    aa2int = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    W_int = torch.full((M, window), -1, dtype=torch.int8, device=device)

    for i, nd in enumerate(tqdm(nodes, desc="  Encoding windows", ncols=80)):
        wstr = nd["window"]
        arr = [aa2int.get(c, -1) for c in wstr]
        if len(arr) < window:
            arr += [-1] * (window - len(arr))
        W_int[i] = torch.tensor(arr, dtype=torch.int8, device=device)

    X = torch.stack(
        [torch.from_numpy(nd["vec"]) for nd in nodes], dim=0
    ).to(device)
    Xn = X / X.norm(dim=1, keepdim=True)
    print("  Tensors ready on", device, flush=True)

    # ── 2) Compute edges (2D blocking) ───────────────────────────────────────
    bs = min(block_size, M)
    block_pairs = [
        (i0, min(i0 + bs, M), j0, min(j0 + bs, M))
        for i0 in range(0, M, bs)
        for j0 in range(i0, M, bs)
    ]

    edges = []
    print(
        f"Computing edges (2D‐blocked) over {len(block_pairs)} block-pairs …",
        flush=True,
    )
    for (i0, i1, j0, j1) in tqdm(
        block_pairs,
        desc="  Computing edges",
        total=len(block_pairs),
        ncols=80,
        leave=True,
    ):
        Xi, Xni, Wi = X[i0:i1], Xn[i0:i1], W_int[i0:i1]
        Xj, Xnj, Wj = X[j0:j1], Xn[j0:j1], W_int[j0:j1]

        eq = (Wi.unsqueeze(1) == Wj.unsqueeze(0)).sum(dim=2).float()
        id_mat = eq / window * 100.0

        cos = Xni @ Xnj.T
        d2 = torch.cdist(Xi, Xj, p=2)
        s2 = 1.0 / (1.0 + d2)
        d1 = torch.cdist(Xi, Xj, p=1)
        s1 = 1.0 / (1.0 + d1)
        comp = (cos + s2 + s1) / 3.0

        mask = (id_mat >= id_thr) | (comp >= comp_thr)
        idxs = mask.nonzero(as_tuple=False)
        for bi, bj in idxs:
            ii = i0 + int(bi.item())
            jj = j0 + int(bj.item())
            if jj > ii:
                edges.append((ii, jj))

        del Xi, Xni, Wi, Xj, Xnj, Wj, eq, id_mat, cos, d2, s2, d1, s1, comp
        del mask, idxs
        torch.cuda.empty_cache()

    # ── 3) Union‐find to group nodes ─────────────────────────────────────────
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    clusters = defaultdict(list)
    for idx, nd in enumerate(nodes:
        ):
        gid = find(idx)
        clusters[gid].append((idx, nd))

    # ── 4) Parallel “Forming clusters” using a process pool ──────────────────
    print("Forming cluster summaries …", flush=True)

    # Prepare a list of (gid, members) pairs for all clusters.
    cluster_items = list(clusters.items())
    # We’ll pass nodes (the full list) by closure to each worker.

    group_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        # Each worker gets (gid, members, nodes)
        futures = []
        for gid, members in cluster_items:
            futures.append(
                exe.submit(_summarize_single_cluster, (gid, members, nodes))
            )

        # As each future completes, append its result
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="  Summarizing clusters",
            ncols=80,
        ):
            group_dict = fut.result()
            group_list.append(group_dict)

    return group_list

