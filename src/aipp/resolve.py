# resolve.py
#!/usr/bin/env python3
"""
Conflict-resolution utilities for the AIPP distiller pipeline.

Major upgrade (2025-04-29):
  - Per-source de-duplication and internal-consistency filtering.
  - Only non-conflicted sources vote in the consensus step.
  - Explicit, deterministic tie-break strategy.
  - 2025-04-29-B: store cb_true_src for downstream --cbseq filter.
"""

from collections import defaultdict
from tqdm import tqdm

# These are set in aipp-distiller.main() before calling into this module.
CONSENSUS_CRITERIA = None  # e.g. (">=", 2)
PRIORITIZE_SOURCES = None  # set([...]) or None
# ---------------------------------------------------------------------------


def _representatives_by_source(records):
    """
    From many raw records, keep one representative per source.
    Discard a source if its own records disagree (internal conflict).
    """
    per_src = defaultdict(list)
    for r in records:
        per_src[r["source"]].append(r)

    good_reps = []
    for rows in per_src.values():
        labels = {r["exp_bin"] for r in rows}
        if len(labels) != 1:  # internally conflicted → discard that source
            continue
        # Pick the single strongest record for this source (highest R)
        good_reps.append(max(rows, key=lambda x: x["R"]))
    return good_reps


# .............................................................................
def _cb_condition(num_true_sources):
    """
    Apply the consensus operator against the threshold.
    Returns True if the condition holds (e.g., num >= 2), else False.
    """
    op, thresh = CONSENSUS_CRITERIA
    if op == "==":
        return num_true_sources == thresh
    elif op == ">":
        return num_true_sources > thresh
    elif op == ">=":
        return num_true_sources >= thresh
    elif op == "<":
        return num_true_sources < thresh
    elif op == "<=":
        return num_true_sources <= thresh
    else:
        return False


# .............................................................................
def resolve_group(records_group):
    """
    Resolve a single (uid, roi) group into one record + an audit report.
    """
    # 0) Quick exit if a prioritized source is present. We pick its top record.
    if PRIORITIZE_SOURCES:
        pr = [r for r in records_group if r["source"] in PRIORITIZE_SOURCES]
        if pr:
            chosen = max(pr, key=lambda r: r["R"])
            chosen["conflict_resolved"] = True
            # Minimal cb_true_src info when fast-pathing
            chosen["cb_true_src"] = 1 if chosen["exp_bin"] else 0
            return chosen, ""

    # 1) Build per-source representatives (only non-conflicted sources survive)
    reps = _representatives_by_source(records_group)
    if not reps:
        # Fallback: if all sources conflicted, keep the single strongest record
        reps = [max(records_group, key=lambda r: r["R"])]

    true_reps = [r for r in reps if r["exp_bin"]]
    false_reps = [r for r in reps if not r["exp_bin"]]
    n_true_src = len({r["source"] for r in true_reps})

    # 2) Apply consensus rule over sources that voted "true"
    if true_reps and _cb_condition(n_true_src):
        winner_label = True
        winner_pool = true_reps
    else:
        winner_label = False
        winner_pool = false_reps if false_reps else reps

    # Pick the strongest record from the chosen pool and set final fields
    chosen = max(winner_pool, key=lambda r: r["R"]).copy()
    chosen["exp_bin"] = winner_label
    chosen["conflict_resolved"] = True
    chosen["cb_true_src"] = n_true_src  # stored for downstream --cbseq

    # 3) Produce a short audit trail for logs
    key = (records_group[0]["uid"], records_group[0]["roi"])
    lines = [f"Conflict for {key[0]}:{key[1]}  →  resolved {winner_label}"]
    lines.append(
        "  non-conflicted sources considered: "
        f"{len(reps)} (TRUE={n_true_src}, FALSE={len(reps)-n_true_src})"
    )
    lines.append(
        f"  chosen record: src={chosen['source']}  R={chosen['R']:.2f}"
    )
    report_block = "\n".join(lines)

    return chosen, report_block


# .............................................................................
def resolve_unambiguous(groups):
    """
    Iterate over all (uid, roi) groups and resolve them one by one.
    Pass through cb_true_src so later steps can use it untouched.
    """
    distilled, conflict_reports = [], []
    for key, rec_list in tqdm(
        groups.items(),
        desc="Resolving conflicts",
        total=len(groups),
        ncols=80,
        ascii=True,
    ):
        resolved, info = resolve_group(rec_list)
        distilled.append(resolved)
        if info:
            conflict_reports.append(info)
    return distilled, conflict_reports


# .............................................................................
def resolve_with_pdbfeats(groups, feats_csv):
    """
    Placeholder hook for structure-based resolution using external features.
    For now, it defers to the existing logic unchanged.
    """
    # Intentional placeholder print kept as in original
    print("YEAH BOY")
    return resolve_unambiguous(groups)


def build_consensus_lookup(group_dict):
    """
    Build a lookup of resolved (uid, roi) pairs.
    Only include entries where all non-conflicted reps agree with the result.
    """
    lookup = {}
    for key, recs in tqdm(
        group_dict.items(),
        desc="Pre-consensus pass",
        total=len(group_dict),
        ncols=80,
        ascii=True,
    ):
        resolved, _ = resolve_group(recs)
        reps = _representatives_by_source(recs)
        if all(r["exp_bin"] == resolved["exp_bin"] for r in reps):
            lookup[key] = resolved
    return lookup


# .............................................................................
def resolve_ambiguous_records(ambiguous_records, consensus_lookup):
    """
    Unfold positional ambiguities using consensus evidence where available.
    Preserve cb_true_src and annotate how the position was decided.
    """
    resolved = {}
    for rec in tqdm(
        ambiguous_records,
        desc="Resolving ambiguities",
        ncols=80,
        ascii=True,
        leave=False,
    ):
        uid = rec["uid"]
        candidates = rec.get("ambiguous_positions", [])
        true_evidence_present = False
        ev_label, ev_R, ev_cb = {}, {}, {}

        # Collect consensus evidence for each candidate position
        for pos in candidates:
            k = (uid, pos)
            if k in consensus_lookup:
                e = consensus_lookup[k]
                ev_label[pos] = e["exp_bin"]
                ev_R[pos] = e["R"]
                ev_cb[pos] = e.get("cb_true_src", 0)
                if e["exp_bin"]:
                    true_evidence_present = True

        # If no true evidence exists anywhere, ignore this ambiguous record
        if not true_evidence_present:
            continue

        # Create per-position records with labels and reasons
        for pos in candidates:
            new_rec = rec.copy()
            new_rec["roi"] = pos

            if pos in ev_label:
                label = ev_label[pos]
                r_val = ev_R[pos]
                cbcnt = ev_cb[pos]
            else:
                label = False
                r_val = -0.5
                cbcnt = 0

            new_rec["exp_bin"] = label
            new_rec["R"] = r_val
            new_rec["cb_true_src"] = cbcnt
            new_rec["ambiguous"] = False
            new_rec["note"] += (
                f"(was_amb:{rec['ambiguous_positions_raw']})"
            )
            new_rec["ambiguous_resolution_reason"] = (
                "Consensus evidence" if pos in ev_label else "Inferred negative"
            )

            key = (uid, pos)
            if key not in resolved or new_rec["R"] > resolved[key]["R"]:
                resolved[key] = new_rec

    return list(resolved.values())

