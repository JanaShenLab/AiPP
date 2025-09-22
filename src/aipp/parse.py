#!/usr/bin/env python3
# parse.py

import os
import glob
import argparse
import re
from tqdm import tqdm

# Set of source codes to skip entirely in load_data()
IGNORE_SOURCES = None


def parse_bool(val):
    """Convert text like 'TRUE' or 'FALSE' to bool."""
    if val.strip().upper() == "TRUE":
        return True
    elif val.strip().upper() == "FALSE":
        return False
    else:
        raise ValueError(f"Cannot convert {val} to boolean.")


def parse_threshold_arg(arg_value):
    """
    Parse thresholds for R, e.g. '>=R', '<=R+0.5'.

    Returns (comp, func), where comp is one of '>','>=','<','<=','=', and
    func is a callable that calculates the numeric threshold from
    (exp_thr, R_val).
    """
    if arg_value is None:
        return None
    m = re.match(r"^(>=|<=|>|<|=)?(.*)$", arg_value.strip())
    if not m:
        raise ValueError(f"Invalid threshold format: {arg_value}")
    comp = m.group(1) or "="
    expr = m.group(2).strip()
    if not expr:
        raise ValueError(f"Missing expression in threshold: {arg_value}")

    def threshold_func(exp_thr, R_val):
        safe_locals = {"R": R_val, "exp_thr": exp_thr}
        try:
            val = eval(expr, {"__builtins__": {}}, safe_locals)
        except Exception as e:
            raise ValueError(f"Error evaluating '{expr}': {e}")
        return float(val)

    return (comp, threshold_func)


def parse_cb_arg(arg_value):
    """
    Parse --cb argument with operators: '==4', '>=2', '1', etc.
    Default operator, if none present, is '>='.
    """
    m = re.match(r"^(==|>=|<=|>|<)?\s*(\d+)$", arg_value.strip())
    if not m:
        raise ValueError(f"Invalid --cb format: {arg_value}")
    op = m.group(1) or ">="
    val = int(m.group(2))
    return (op, val)


def load_data(directory):
    """
    Load records from 'lcr_*.dat'.

    Each file row: uid, roi, R, exp_thr, exp_bin, source, note, [mw].
    Multi-value vs. ambiguous is determined by splitting fields with
    semicolons. Records are appended to a big list.
    """
    pattern = os.path.join(directory, "lcr_*.dat")
    files = glob.glob(pattern)

    # Skip whole files whose code is in IGNORE_SOURCES
    if IGNORE_SOURCES:
        kept = []
        for fn in files:
            base = os.path.basename(fn)
            if base.startswith("lcr_") and base.endswith(".dat"):
                code = base[len("lcr_") : -len(".dat")]
                if code in IGNORE_SOURCES:
                    continue
            kept.append(fn)
        files = kept

    all_records = []
    for filename in tqdm(
        files, desc="Parsing input files", ncols=80, ascii=True
    ):
        with open(filename, "r") as f:
            lines = f.readlines()
        start_idx = 0
        if lines and lines[0].lower().startswith("uid"):
            start_idx = 1

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            (
                uid_field,
                roi_field,
                R_field,
                exp_thr_field,
                exp_bin_field,
                source_field,
                note_field,
            ) = parts[:7]

            uid_vals = [x.strip() for x in uid_field.split(";") if x.strip()]
            roi_vals = [x.strip() for x in roi_field.split(";") if x.strip()]
            num_uid, num_roi = len(uid_vals), len(roi_vals)

            # Decide ambiguous vs multi-value
            if num_uid > 1 or num_roi > 1:
                if ((num_uid > 1 and num_roi == 1) or (num_uid == num_roi)):
                    ambiguous_flag = False
                    multi_value = True
                else:
                    ambiguous_flag = True
                    multi_value = False
            else:
                ambiguous_flag = False
                multi_value = False

            # Parse numeric fields
            try:
                R_value = float(R_field)
                exp_thr_value = float(exp_thr_field)
                exp_bin_value = parse_bool(exp_bin_field)
            except Exception:
                continue

            record = {}
            # Multi-value: take first ROI; keep all candidates in metadata
            if multi_value:
                record["uid"] = uid_vals[0]
                try:
                    first_roi = int(roi_vals[0])
                except Exception:
                    continue
                record["roi"] = first_roi
                record["ambiguous_positions_raw"] = roi_field.strip()
                try:
                    record["ambiguous_positions"] = [int(x) for x in roi_vals]
                except Exception:
                    record["ambiguous_positions"] = []

            # Ambiguous: cannot pair up UID/ROI neatly
            elif ambiguous_flag:
                record["uid"] = uid_vals[0]
                record["ambiguous_positions_raw"] = roi_field.strip()
                try:
                    pos_list = [int(x) for x in roi_vals]
                except Exception:
                    pos_list = []
                record["ambiguous_positions"] = pos_list
                # Provide a 'roi' anyway so build_cys_groups won’t KeyError
                if pos_list:
                    record["roi"] = pos_list[0]
                else:
                    # If we cannot parse any int, skip entirely
                    continue

            # Unambiguous single-value
            else:
                record["ambiguous_positions"] = None
                record["ambiguous_positions_raw"] = None
                record["uid"] = uid_field.strip()
                try:
                    record["roi"] = int(roi_field.strip())
                except Exception:
                    continue

            # Common fields
            record["R"] = R_value
            record["exp_thr"] = exp_thr_value
            record["exp_bin"] = exp_bin_value
            record["source"] = source_field.strip()
            record["note"] = note_field.strip()
            record["mw"] = (
                float(parts[7])
                if (
                    len(parts) >= 8
                    and parts[7].replace(".", "", 1).isdigit()
                )
                else None
            )
            record["conflict_resolved"] = False
            record["ambiguous"] = ambiguous_flag
            record["multi_value"] = multi_value
            record["og_uid"] = record["uid"]

            all_records.append(record)

    return all_records


def parse_arguments():
    """
    Set up argparse for command-line usage. Return parsed args.
    """
    p = argparse.ArgumentParser(
        description="Distill dataset from lcr_*.dat."
    )
    # Existing options
    p.add_argument(
        "--dir",
        required=True,
        help="Directory with lcr_*.dat files",
    )
    p.add_argument(
        "--neg-max-r",
        type=str,
        default=None,
        help="Example: '<=R', '<R/2', etc.",
    )
    p.add_argument(
        "--pos-min-r",
        type=str,
        default=None,
        help="Example: '>=R', '>R+0.5', etc.",
    )
    p.add_argument(
        "--neg-min-r",
        type=str,
        default=None,
        help="Example: '>=R+0.5', '=1.2'.",
    )
    p.add_argument(
        "--pos-max-r",
        type=str,
        default=None,
        help="Example: '<=R', '<R*2'.",
    )
    p.add_argument(
        "--cb",
        type=str,
        default="1",
        help="Consensus threshold (e.g. '==4','>1','>=2').",
    )
    p.add_argument(
        "--cbseq",
        type=str,
        default=None,
        help=(
            "Protein-level filter: keep only proteins that contain ≥1 "
            "positive meeting this CB rule"
        ),
    )
    p.add_argument(
        "--allownegonly",
        action="store_true",
        help="Retain negative records even w/o positives.",
    )
    p.add_argument(
        "--rmax",
        type=float,
        default=None,
        help="If set, clamp R>rmax to rmax+0.01.",
    )
    p.add_argument(
        "--maxMW",
        type=float,
        default=float(\"inf\"),
        help="Max MW filter.",
    )
    p.add_argument(
        "--minMW",
        type=float,
        default=0,
        help="Min MW filter.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    p.add_argument(
        "--usepdbs",
        action="store_true",
        help="Use mapped pdb data for conflict resolution.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude sources or .dat of uids.",
    )
    p.add_argument(
        "--ignore",
        type=str,
        default=None,
        help="Comma-separated list of source codes to ignore.",
    )
    p.add_argument(
        "--prioritize",
        type=str,
        default=None,
        help="Comma list of sources to prefer.",
    )
    p.add_argument(
        "--unconflictedonly",
        action="store_true",
        help=(
            "Filter out points with conflicting TRUE/FALSE labels."
        ),
    )
    p.add_argument(
        "--unambiguousonly",
        action="store_true",
        help="Filter out records with ambiguous UID or ROI fields.",
    )
    p.add_argument(
        "--pdbfeats",
        metavar="CSV",
        default=None,
        help="Path to CSV of structure-derived features",
    )

    # New cysteine-grouping options
    p.add_argument(
        "--seqs",
        type=str,
        default=None,
        help="Multi-FASTA file for primary sequences",
    )
    p.add_argument(
        "--repr",
        type=str,
        default="0",
        help="Comma-separated ESM layers to use from .pt files",
    )
    p.add_argument(
        "--embs",
        type=str,
        default=None,
        help="Directory of per-protein .npy embeddings",
    )
    p.add_argument(
        "--window",
        type=int,
        default=21,
        help="Window size centered on residue (odd int)",
    )
    p.add_argument(
        "--identitythr",
        type=float,
        default=70.0,
        help="Sequence identity threshold (percent)",
    )
    p.add_argument(
        "--compthr",
        type=float,
        default=0.3,
        help="Composite embedding similarity threshold",
    )
    p.add_argument(
        "--nmlb",
        type=str,
        default=None,
        help=(
            "Comma-separated single-letter residues to add as FALSE-"
            "labelled “not modelled but labellable” (e.g. 'C' or 'K,H'); "
            "requires --usegroups and --prioritize."
        ),
    )
    p.add_argument(
        "--filterfor",
        type=str,
        default=None,
        help=(
            "Only emit entries whose UID appears in the specified source’s "
            "input file. UIDs added via --nmlb will still pass if their "
            "protein was in that source."
        ),
    )

    # New group-level resolution flags
    p.add_argument(
        "--usegroups",
        action="store_true",
        help=(
            "Perform group-level conflict resolution instead of "
            "per-residue."
        ),
    )
    p.add_argument(
        "--repsonly",
        action="store_true",
        help=(
            "When --usegroups: emit only the group representatives in the "
            "final dataset."
        ),
    )

    return p.parse_args()


def setup_thresholds(args):
    """
    Parse threshold strings into (op, threshold_func) forms.

    Return them plus a dict of operator lambdas used to compare values.
    """
    neg_max = (
        parse_threshold_arg(args.neg_max_r) if args.neg_max_r else None
    )
    pos_min = (
        parse_threshold_arg(args.pos_min_r) if args.pos_min_r else None
    )
    neg_min = (
        parse_threshold_arg(args.neg_min_r) if args.neg_min_r else None
    )
    pos_max = (
        parse_threshold_arg(args.pos_max_r) if args.pos_max_r else None
    )

    if neg_max is not None:
        comp, fn = neg_max
        if comp == "=":
            comp = "<="
        neg_max_comp, neg_max_func = comp, fn
    else:
        neg_max_comp, neg_max_func = None, None

    if pos_min is not None:
        comp, fn = pos_min
        if comp == "=":
            comp = ">="
        pos_min_comp, pos_min_func = comp, fn
    else:
        pos_min_comp, pos_min_func = None, None

    if neg_min is not None:
        comp, fn = neg_min
        if comp == "=":
            comp = ">="
        neg_min_comp, neg_min_func = comp, fn
    else:
        neg_min_comp, neg_min_func = None, None

    if pos_max is not None:
        comp, fn = pos_max
        if comp == "=":
            comp = "<="
        pos_max_comp, pos_max_func = comp, fn
    else:
        pos_max_comp, pos_max_func = None, None

    comp_ops = {
        ">=": lambda x, y: x >= y,
        ">": lambda x, y: x > y,
        "<=": lambda x, y: x <= y,
        "<": lambda x, y: x < y,
        "=": lambda x, y: x == y,
    }

    return (
        pos_min_comp,
        pos_min_func,
        pos_max_comp,
        pos_max_func,
        neg_max_comp,
        neg_max_func,
        neg_min_comp,
        neg_min_func,
        comp_ops,
    )

