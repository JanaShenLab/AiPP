#!/usr/bin/env python3

"""
Group-level conflict resolution for cysteine clusters.

Goal:
- Decide a single label for each group by "consensus-by-count" (CB).
- Add a parameter `novote_sources`: sources that stay in the group but
  do not cast TRUE/FALSE votes.

Key ideas:
- Records from `novote_sources` are included in the output and inherit the
  final group label, but they do not affect the vote counts.
- Optional `prioritize_sources` can force a group TRUE if any of those
  sources vote TRUE.
- Separate minimum-source requirements for positive and negative calls.

Terminology:
- "TRUE" / "FALSE" are member labels that count toward voting.
- "MASK" means we could not confidently call TRUE or FALSE.
"""

from operator import eq, gt, ge, lt, le
from collections import defaultdict  # noqa: F401 (imported, kept for context)

# Mapping from operator text to Python functions
_OP_FUNCS = {
    "==": eq,
    ">": gt,
    ">=": ge,
    "<": lt,
    "<=": le,
}


def resolve_groups(
    groups,
    cb_rule_pos,
    cb_rule_neg=None,
    prioritize_sources=None,
    novote_sources=None,
    min_src_per_pos=1,
    min_src_per_neg=1,
):
    """
    Args:
      groups:
        list of dicts like:
          {
            'group_id': ...,
            'representative': {'uid', 'roi'},
            'members': [
              {'uid','roi','label','source','R', ...},
              ...
            ]
          }

      cb_rule_pos:
        Tuple (op, threshold) for positive voting. Example: (">=", 2).

      cb_rule_neg:
        Optional (op, threshold) for negative voting. If None, use the
        positive rule for negatives too.

      prioritize_sources:
        Set of source codes; if any of these has a TRUE record, the group
        becomes TRUE immediately.

      novote_sources:
        Set of source codes whose records DO NOT vote (but remain members).

      min_src_per_pos / min_src_per_neg:
        Minimum number of distinct voting sources required to call TRUE
        or FALSE, respectively. If unmet, result is "MASK".

    Returns:
      list of dicts with keys:
        'group_id','uid','roi','R','label',
        'true_count','false_count','members'

      Notes:
        - 'label' ∈ {True, False, "MASK"} based on the CB rules.
        - Members from `novote_sources` do not contribute to true/false
          counts, but they inherit the group's final label.
    """
    # --- parse/validate positive CB rule ---
    op_pos, thresh_pos = cb_rule_pos
    if op_pos not in _OP_FUNCS:
        raise ValueError(f"Unsupported CB operator for positives: {op_pos}")
    cb_func_pos = _OP_FUNCS[op_pos]

    # --- parse/validate negative CB rule (fallback to positive if not given) ---
    if cb_rule_neg is None:
        cb_func_neg = cb_func_pos
        op_neg, thresh_neg = op_pos, thresh_pos
    else:
        op_neg, thresh_neg = cb_rule_neg
        if op_neg not in _OP_FUNCS:
            raise ValueError(f"Unsupported CB operator for negatives: {op_neg}")
        cb_func_neg = _OP_FUNCS[op_neg]

    # Normalize inputs to sets for quick membership checks
    prioritized = set(prioritize_sources) if prioritize_sources else set()
    novote = set(novote_sources) if novote_sources else set()

    resolved = []
    for g in groups:
        members = g["members"]

        # --- Count votes from sources NOT in novote_sources ---
        # Only literal "TRUE"/"FALSE" labels count toward these tallies.
        true_count = 0
        false_count = 0
        for m in members:
            src = m["source"]
            # Some pipelines carry a per-record "note"; respect novote if it
            # appears there too (kept behavior).
            note = m["note"]
            if src in novote or note in novote:
                continue
            if m["label"] == "TRUE":
                true_count += 1
            elif m["label"] == "FALSE":
                false_count += 1

        # --- Decide the preliminary group label ---
        # 1) If any prioritized source has a TRUE, force group TRUE.
        if prioritized and any(
            (m["source"] in prioritized and m["label"] == "TRUE")
            for m in members
        ):
            group_label = True
        else:
            # 2a) Try positive rule against the TRUE count.
            if cb_func_pos(true_count, thresh_pos):
                group_label = True
            # 2b) If there were zero TRUE votes, try negative rule.
            elif true_count == 0 and cb_func_neg(false_count, thresh_neg):
                group_label = False
            else:
                group_label = "MASK"

        # --- Enforce minimum distinct-source requirements per side ---
        if group_label is True:
            if min_src_per_pos > 1:
                supporting = [
                    m
                    for m in members
                    if m["source"] not in novote and m["label"] == "TRUE"
                ]
                srcs = {m["source"] for m in supporting}
                # If no prioritized voter and too few sources, mask it.
                if not (srcs & prioritized) and len(srcs) < min_src_per_pos:
                    group_label = "MASK"

        elif group_label is False:
            if min_src_per_neg > 1:
                supporting = [
                    m
                    for m in members
                    if m["source"] not in novote and m["label"] == "FALSE"
                ]
                srcs = {m["source"] for m in supporting}
                if len(srcs) < min_src_per_neg:
                    group_label = "MASK"

        # --- Find representative's R value (keeps representative metadata) ---
        rep = g["representative"]
        rep_R = None
        for m in members:
            if m["uid"] == rep["uid"] and m["roi"] == rep["roi"]:
                rep_R = m["R"]
                break

        # --- Build the resolved record for this group ---
        resolved.append(
            {
                "group_id": g["group_id"],
                "uid": rep["uid"],
                "roi": rep["roi"],
                "R": rep_R,
                "label": group_label,
                "true_count": true_count,
                "false_count": false_count,
                "members": members,
            }
        )

    return resolved

