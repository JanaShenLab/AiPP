# report.py
"""
Reporting utilities for the AIPP distiller pipeline:
  - compute_stats: basic statistics on numeric lists
  - generate_statistics: overall and per-source R/value summaries
  - generate_ambiguity_report: text blocks for ambiguous entries
  - write_report: dump a report string to a file
"""

import statistics
from collections import defaultdict


def compute_stats(values):
    """
    Compute min, max, range, mean, and stddev for a list of floats.

    For empty input, return zeros so callers do not need special cases.
    """
    if not values:
        return {"min": 0, "max": 0, "range": 0, "mean": 0, "stddev": 0}
    mn = min(values)
    mx = max(values)
    rng = mx - mn
    mean_ = sum(values) / len(values)
    std_ = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"min": mn, "max": mx, "range": rng, "mean": mean_, "stddev": std_}


def generate_statistics(records):
    """
    Summarize stats overall and per source.

    Returns:
      report (str): multi-line statistical report
      src_counts (dict): source -> {"total":…, "TRUE":…, "FALSE":…}
    """
    total = len(records)
    amb_raw = sum(
        1
        for r in records
        if r.get("ambiguous", False) or r.get("multi_value", False)
    )
    true_cnt = sum(1 for r in records if r["exp_bin"])
    false_cnt = sum(1 for r in records if not r["exp_bin"])
    tot_bin = true_cnt + false_cnt
    pct_true = (true_cnt / tot_bin * 100) if tot_bin else 0
    pct_false = (false_cnt / tot_bin * 100) if tot_bin else 0

    all_R = [r["R"] for r in records]
    true_R = [r["R"] for r in records if r["exp_bin"]]
    false_R = [r["R"] for r in records if not r["exp_bin"]]

    s_all = compute_stats(all_R)
    s_true = compute_stats(true_R)
    s_false = compute_stats(false_R)

    # Collect per-source values and counts.
    src_all = defaultdict(list)
    src_true = defaultdict(list)
    src_false = defaultdict(list)
    src_counts = defaultdict(lambda: {"total": 0, "TRUE": 0, "FALSE": 0})

    for r in records:
        src = r["source"]
        src_all[src].append(r["R"])
        src_counts[src]["total"] += 1
        if r["exp_bin"]:
            src_true[src].append(r["R"])
            src_counts[src]["TRUE"] += 1
        else:
            src_false[src].append(r["R"])
            src_counts[src]["FALSE"] += 1

    # Build a readable, plain-text report.
    lines = []
    lines.append("STATISTICAL REPORT")
    lines.append("==================")
    lines.append(f"Total records: {total}")
    lines.append(f"Ambiguous/multi-value count: {amb_raw}")
    lines.append("")
    lines.append("exp_bin breakdown:")
    lines.append(f"  TRUE:  {true_cnt} ({pct_true:.2f}%)")
    lines.append(f"  FALSE: {false_cnt} ({pct_false:.2f}%)")
    lines.append("")
    lines.append("R-value stats:")
    lines.append("  ALL:")
    lines.append(f"    Min:   {s_all['min']:.2f}")
    lines.append(f"    Max:   {s_all['max']:.2f}")
    lines.append(f"    Range: {s_all['range']:.2f}")
    lines.append(f"    Mean:  {s_all['mean']:.2f}")
    lines.append(f"    Stdev: {s_all['stddev']:.2f}")
    lines.append("  TRUE only:")
    lines.append(f"    Min:   {s_true['min']:.2f}")
    lines.append(f"    Max:   {s_true['max']:.2f}")
    lines.append(f"    Range: {s_true['range']:.2f}")
    lines.append(f"    Mean:  {s_true['mean']:.2f}")
    lines.append(f"    Stdev: {s_true['stddev']:.2f}")
    lines.append("  FALSE only:")
    lines.append(f"    Min:   {s_false['min']:.2f}")
    lines.append(f"    Max:   {s_false['max']:.2f}")
    lines.append(f"    Range: {s_false['range']:.2f}")
    lines.append(f"    Mean:  {s_false['mean']:.2f}")
    lines.append(f"    Stdev: {s_false['stddev']:.2f}")
    lines.append("")

    def produce_table(dct, count_dct, total_for_cat, pct_label):
        """
        Create a tab-separated summary table for a category (ALL/TRUE/FALSE).
        """
        hdr = "\t".join(
            [
                "Source".ljust(12),
                "Count".rjust(12),
                pct_label.rjust(12),
                "R_Min".rjust(7),
                "R_Max".rjust(7),
                "R_Rng".rjust(7),
                "R_Mean".rjust(7),
                "R_Std".rjust(7),
            ]
        )
        tbl = [hdr]
        all_srcs = set(src_all.keys())
        for src in sorted(all_srcs):
            vals = dct.get(src, [])
            cnt = count_dct.get(src, 0)
            pct = (cnt / total_for_cat * 100) if total_for_cat else 0
            st = (
                compute_stats(vals)
                if vals
                else {"min": 0, "max": 0, "range": 0, "mean": 0, "stddev": 0}
            )
            row = "\t".join(
                [
                    f"{src:<12}",
                    f"{len(vals):12d}",
                    f"{pct:11.2f}%",
                    f"{st['min']:7.2f}",
                    f"{st['max']:7.2f}",
                    f"{st['range']:7.2f}",
                    f"{st['mean']:7.2f}",
                    f"{st['stddev']:7.2f}",
                ]
            )
            tbl.append(row)
        return "\n".join(tbl)

    tot_all = sum(src_counts[s]["total"] for s in src_counts)
    tot_true = sum(src_counts[s]["TRUE"] for s in src_counts)
    tot_false = sum(src_counts[s]["FALSE"] for s in src_counts)

    lines.append("Per-source (ALL):")
    lines.append(
        produce_table(
            src_all,
            {s: src_counts[s]["total"] for s in src_counts},
            tot_all,
            "%Total",
        )
    )
    lines.append("")
    lines.append("Per-source (TRUE):")
    lines.append(
        produce_table(
            src_true,
            {s: src_counts[s]["TRUE"] for s in src_counts},
            tot_true,
            "%TRUE",
        )
    )
    lines.append("")
    lines.append("Per-source (FALSE):")
    lines.append(
        produce_table(
            src_false,
            {s: src_counts[s]["FALSE"] for s in src_counts},
            tot_false,
            "%FALSE",
        )
    )

    report = "\n".join(lines)
    return report, src_counts


def generate_ambiguity_report(ambiguous_records, distilled_unambig_dict):
    """
    For each ambiguous record, show candidates and any evidence present.

    Returns a list of text blocks (one per ambiguous record).
    """
    reports = []
    for rec in ambiguous_records:
        uid = rec["uid"]
        block = []
        block.append(f"Ambiguous entry uid='{uid}'")
        block.append(
            f"  Original ROI field: {rec['ambiguous_positions_raw']}"
        )
        candidates = rec.get("ambiguous_positions", [])
        if not candidates:
            block.append("  No valid candidate positions parsed.")
            block.append("  Overall Resolution: UNRESOLVED")
        else:
            unresolved = False
            for pos in candidates:
                key = (uid, pos)
                if key in distilled_unambig_dict:
                    e = distilled_unambig_dict[key]
                    block.append(
                        f"  Cand pos {pos:4d}: {e['exp_bin']}  "
                        f"(src={e['source']}, R={e['R']:.2f}, "
                        f"note={e['note']})"
                    )
                else:
                    block.append(
                        f"  Cand pos {pos:4d}: NO evidence found"
                    )
                    unresolved = True

            any_true = any(
                distilled_unambig_dict.get((uid, pos), {}).get("exp_bin")
                for pos in candidates
            )
            if unresolved or not any_true:
                block.append("  Overall: UNRESOLVED (lack of TRUE evidence)")
            else:
                block.append("  Overall: RESOLVED (evidence found)")

        block.append(f"  Original Source: {rec['source']}")
        reports.append("\n".join(block))

    return reports


def write_report(report_text, filename):
    """
    Write the given report_text (string) to filename, appending a newline.

    Simple helper to keep file writes consistent across the codebase.
    """
    with open(filename, "w") as f:
        f.write(report_text + "\n")

