#!/usr/bin/env python3
"""
Grouping‐related utilities:

  • filter_by_length
  • build_needed
  • find_missing
  • build_nodes
  • cluster_nodes
  • write_missing_fasta

Key principle:

  • We cluster based on one “node” per unique (uid, mapped_roi).  But we
    keep a list of all original records at that same (uid, mapped_roi).
    That way, duplicates appear in the final report, but do not skew
    centroid selection.
"""

import os
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------------------
# If True, we never store or use any sequence windows (window_seq),
# skipping all subsequence‐identity logic.
# This switch lets you quickly disable window/identity features.
_SKIP_SEQ_WINDOWS = False
# -------------------------------------------------------------------


def filter_by_length(records, minlen, maxlen):
    """
    Discard any record whose full sequence length is < minlen or > maxlen.
    Returns a new list of passing records.
    """
    out = []
    for r in records:
        seq = r.get("sequence", "")
        if not seq:
            continue
        L = len(seq)
        if L < minlen or L > maxlen:
            continue
        out.append(r)
    return out


def build_needed(records):
    """
    Build a dict mapping embedding_key → set of needed ROI positions.

    For each record r:
      r["uid"] is exactly the filename (without extension) of the .pt/.npy in
      <emb_dir>, r["mapped_roi"] is the 1-based index in that sequence to
      request.

    We register one entry per record (duplicates will simply add the same pos
    to the set, but sets remove duplicates naturally).
    """
    needed = defaultdict(set)
    for r in records:
        uid = r["uid"]              # PDB‐chain or UniProt, from load_dataset
        pos = r["mapped_roi"]       # 1-based index in r["sequence"]
        needed[uid].add(pos)
    return needed


def find_missing(records, embs):
    """
    Compare each record r against `embs`: { uid: { mapped_roi → vector } }.

    For each r:
      emb_map = embs.get(r["uid"])
      if emb_map is None or r["mapped_roi"] not in emb_map,
        then r is missing.

    Returns:
      missing_records: list of all such r
      uniq_cys:        set of (r["uid"], r["mapped_roi"])
      uniq_uids:       set of r["uid"]
    """
    missing_records = []
    for r in records:
        uid = r["uid"]
        pos = r["mapped_roi"]

        emb_map = embs.get(uid)
        if emb_map is None or pos not in emb_map:
            missing_records.append(r)

    uniq_cys = {(r["uid"], r["mapped_roi"]) for r in missing_records}
    uniq_uids = {r["uid"] for r in missing_records}
    return missing_records, uniq_cys, uniq_uids


class _Node:
    """
    Internal helper: represent one “unique (uid, mapped_roi)” with:
      - uid, pos
      - window_seq (all duplicates share the same)
      - vec (embedding vector)
      - rec_list = list of all original records mapped here
    """
    __slots__ = ("uid", "pos", "window", "vec", "rec_list")

    def __init__(self, uid, pos, window, vec):
        self.uid = uid
        self.pos = pos
        self.window = window
        self.vec = vec
        # We’ll append all records with this (uid, pos)
        self.rec_list = []


def build_nodes(records, embs):
    """
    Build exactly one “_Node” per unique (uid, mapped_roi), but collect all
    original records under that node’s rec_list.

    Steps:
      1) Index all records by (uid, mapped_roi).
      2) For each unique (uid, pos):
         a) r0 = first record in that group
         b) emb_map = embs.get(uid); if missing or pos not in emb_map, skip.
         c) vec = emb_map[pos]
         d) wstr = r0["window_seq"]
         e) Create node = _Node(uid, pos, wstr, vec)
         f) node.rec_list = [ all records with that (uid, pos) ]
      3) Return list of all node instances.
    """
    # 1) Group records by (uid, mapped_roi)
    by_cys = defaultdict(list)
    for r in records:
        key = (r["uid"], r["mapped_roi"])
        by_cys[key].append(r)

    nodes = []
    for (uid, pos), rec_list in by_cys.items():
        # 2a) pick the first record to fetch window_seq
        r0 = rec_list[0]
        emb_map = embs.get(uid)
        if emb_map is None or pos not in emb_map:
            continue

        # 2c) fetch vector
        vec = emb_map[pos]

        # 2d) fetch window_seq (or skip entirely)
        wstr = "" if _SKIP_SEQ_WINDOWS else r0.get("window_seq", "")
        # Drop only if vec is missing; allow empty wstr when skipping
        if vec is None:
            continue

        # 2e) new node
        node = _Node(uid, pos, wstr, vec)
        # 2f) attach all duplicates
        node.rec_list = rec_list.copy()
        nodes.append(node)

    return nodes


def cluster_nodes(nodes, window, id_thr, comp_thr, block_size=512):
    """
    Perform 2D‐blocked GPU clustering on `nodes` (one node per unique
    (uid,pos)).

    Returns a list of clusters, each a dict:
      {
        "group_id": int,
        "representative": {"uid":…, "roi":…},
        "members": [
           {
             "uid": r["uid"],         # identical across duplicates
             "roi": r["mapped_roi"],  # identical across duplicates
             "identity": …,
             "comp": …,
             "label": "TRUE"/"FALSE",
             "source": r["source"],
             "note": r.get("note",""),
             "R": r["R"]
           }
           for every r in rec_list of every node in this cluster
        ]
      }

    Internally:
      • We still treat each node as a single vector/window for identity+comp
        (duplicates do not increase voting power).
      • Once clusters form, we choose a representative node (highest avg comp
        to other nodes in that cluster).  Then we expand **all** rec_list
        entries under **every** node in that cluster.  That way duplicates
        appear in the final “members” list.
    """
    M = len(nodes)
    if M == 0:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    # ── Optionally build W_int windows (only if we’ll use seq‐identity) ─
    use_seqid = not _SKIP_SEQ_WINDOWS and (id_thr <= 100.0)
    if use_seqid:
        aa2int = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        W_int = torch.full(
            (M, window),
            -1,
            dtype=torch.int8,
            device=device
        )
        for i, node in enumerate(
            tqdm(nodes, desc="  Encoding windows", ncols=80)
        ):
            arr = [aa2int.get(c, -1) for c in node.window]
            if len(arr) < window:
                arr += [-1] * (window - len(arr))
            W_int[i] = torch.tensor(arr, dtype=torch.int8, device=device)

    X = torch.stack(
        [torch.from_numpy(node.vec) for node in nodes],
        dim=0
    ).to(device)
    Xn = X / X.norm(dim=1, keepdim=True)
    print("  Tensors ready on", device, flush=True)

    # ── Build block_pairs ───────────────────────────────────────────
    bs = min(block_size, M)
    block_pairs = [
        (i0, min(i0 + bs, M), j0, min(j0 + bs, M))
        for i0 in range(0, M, bs)
        for j0 in range(i0, M, bs)
    ]

    # ── Compute edges (2D‐blocked) ───────────────────────────────────
    edges = []
    print(
        f"Computing edges over {len(block_pairs)} block‐pairs …",
        flush=True
    )
    for (i0, i1, j0, j1) in tqdm(
        block_pairs,
        desc="  Computing edges",
        total=len(block_pairs),
        ncols=80,
        leave=True
    ):
        Xi, Xni = X[i0:i1], Xn[i0:i1]
        Xj, Xnj = X[j0:j1], Xn[j0:j1]

        # 1) sequence‐identity if enabled
        if use_seqid:
            Wi = W_int[i0:i1]
            Wj = W_int[j0:j1]
            eq = (Wi.unsqueeze(1) == Wj.unsqueeze(0)).sum(dim=2).float()
            id_mat = eq / window * 100.0
            seq_mask = (id_mat >= id_thr)
        else:
            seq_mask = torch.zeros(
                (i1 - i0, j1 - j0),
                dtype=torch.bool,
                device=device
            )

        # 2) composite embedding similarity
        cos = Xni @ Xnj.T
        d2 = torch.cdist(Xi, Xj, p=2)
        s2 = 1.0 / (1.0 + d2)
        d1 = torch.cdist(Xi, Xj, p=1)
        s1 = 1.0 / (1.0 + d1)
        comp = (cos + s2 + s1) / 3.0

        # 3) final edge mask: either seq_id or comp
        mask = seq_mask | (comp >= comp_thr)

        idxs = mask.nonzero(as_tuple=False)
        for bi, bj in idxs:
            ii = i0 + int(bi.item())
            jj = j0 + int(bj.item())
            if jj > ii:
                edges.append((ii, jj))

        del Xi, Xni, Xj, Xnj, cos, d2, s2, d1, s1, comp, mask, idxs
        if use_seqid:
            del Wi, Wj, eq, id_mat
        torch.cuda.empty_cache()

    # ── Union‐Find to cluster ───────────────────────────────────────────
    parent = list(range(M))

    def find(x):
        # Standard path compression for speed
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
    for idx, node in enumerate(nodes):
        gid = find(idx)
        clusters[gid].append((idx, node))

    # ── Build group_list with representatives ─────────────────────────
    group_list = []
    print("Forming cluster summaries …", flush=True)
    for gid, member_nodes in tqdm(
        clusters.items(),
        desc="  Summarizing",
        ncols=80
    ):
        # 1) Find representative node by highest avg comp to other nodes
        sims = {}
        for idx, node in member_nodes:
            tot_sim, cnt = 0.0, 0
            for jdx, other_node in member_nodes:
                if idx == jdx:
                    continue
                vi = node.vec / np.linalg.norm(node.vec)
                vj = other_node.vec / np.linalg.norm(other_node.vec)
                tot_sim += float(np.dot(vi, vj))
                cnt += 1
            sims[idx] = (tot_sim / cnt) if cnt else 0.0

        rep_idx = max(sims, key=sims.get)
        rep_node = nodes[rep_idx]
        # ── FIXED: use "roi" key here, not "pos"
        rep = {"uid": rep_node.uid, "roi": rep_node.pos}

        # 2) Expand every original record under every node in this cluster
        items = []
        rep_w, rep_v = rep_node.window, rep_node.vec
        for idx, node in member_nodes:
            for rec in node.rec_list:  # include duplicates explicitly
                items.append({
                    "uid": rec["uid"],
                    "roi": rec["mapped_roi"],
                    "identity": _seq_identity(rep_w, node.window),
                    "comp": _comp_similarity(rep_v, node.vec),
                    "label": "TRUE" if rec["exp_bin"] else "FALSE",
                    "source": rec["source"],
                    "note": rec.get("note", ""),
                    "R": rec["R"],
                })

        group_list.append({
            "group_id": gid,
            "representative": rep,
            "members": items,
        })

    return group_list


def _seq_identity(w1, w2):
    """
    Percent identity between two windows w1, w2.
    """
    L = min(len(w1), len(w2))
    if L == 0:
        return 0.0
    matches = sum(1 for a, b in zip(w1[:L], w2[:L]) if a == b)
    return matches / L * 100.0


def _comp_similarity(v1, v2):
    """
    Composite embedding similarity = (cosine + s2 + s1) / 3.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    vi = v1 / n1
    vj = v2 / n2
    cos_ij = float(np.dot(vi, vj))
    diff = v1 - v2
    d2 = float(np.linalg.norm(diff, ord=2))
    d1 = float(np.linalg.norm(diff, ord=1))
    s2 = 1.0 / (1.0 + d2)
    s1 = 1.0 / (1.0 + d1)
    return (cos_ij + s2 + s1) / 3.0


def write_missing_fasta(missing_uids, seqs_fasta_path, report_dir):
    """
    Write “missing.fasta” containing full sequences for each UID in
    missing_uids, by re‐reading seqs_fasta_path (the 'uid_sequences.fasta'
    that load_dataset wrote).

    missing_uids: set of UIDs (exact strings)
    seqs_fasta_path: path to existing 'uid_sequences.fasta'
    report_dir: directory in which to write 'missing.fasta'
    """
    uid_to_seq = {}
    with open(seqs_fasta_path) as f:
        current_uid = None
        current_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_uid:
                    uid_to_seq[current_uid] = "".join(current_lines)
                hdr = line[1:].split()[0]
                current_uid = hdr
                current_lines = []
            else:
                current_lines.append(line)
        if current_uid:
            uid_to_seq[current_uid] = "".join(current_lines)

    out_path = os.path.join(report_dir, "missing.fasta")
    with open(out_path, "w") as outf:
        for uid in sorted(missing_uids):
            seq = uid_to_seq.get(uid)
            if seq:
                outf.write(f">{uid}\n{seq}\n")

    print(
        f"Missing‐embeddings FASTA written to: {out_path}",
        flush=True
    )

