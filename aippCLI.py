#!/usr/bin/env python3

import argparse
import json
import os
from typing import List, Tuple, Dict, Any
import time
import random
import sys
import shlex

import numpy as np
import torch
import torch.nn as nn
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
import getpass
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
#  Globals: default weights location + Forge token cache
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WTS_ROOT = os.environ.get(
    "AIPP_WTS_DIR",
    os.path.join(HERE, "env/wts"),
)

TOKEN_FILE = os.environ.get(
    "AIPP_FORGE_TOKEN_FILE",
    os.path.join(os.path.expanduser("~"), ".aipp_forge_token"),
)


def load_saved_token(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            token = f.read().strip()
        return token or None
    except FileNotFoundError:
        return None


def save_token(path: str, token: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(token.strip() + "\n")


# ---------------------------------------------------------------------------
#  1. Config loader
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> Dict[str, Any]:
    """
    Load the run.cfg JSON (assumes valid JSON like the one you pasted).
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


# ---------------------------------------------------------------------------
#  2. Model definition
# ---------------------------------------------------------------------------

class MLPHead2(nn.Module):
    """
    Simple MLP head:
      - One LayerNorm up front
      - `cfg['num_layers']` of (Linear → GELU), each with its own width
      - One Dropout
      - Final Linear to out_dim
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg['emb_dim']
        hid_dim = cfg['hid_dim']
        out_dim = cfg['out_dim']
        n_layers = cfg['num_layers']
        dropout_p = cfg['dropout']

        # normalize once
        self.norm = nn.LayerNorm(emb_dim)

        # interpret hid_dim
        if isinstance(hid_dim, (list, tuple)):
            if len(hid_dim) != n_layers:
                raise ValueError(
                    "Expected hid_dim list of length "
                    f"{n_layers}, got {len(hid_dim)}"
                )
            hid_dims = list(hid_dim)
        else:
            hid_dims = [hid_dim] * n_layers

        # build hidden stack
        layers = []
        in_dim = emb_dim
        for h in hid_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h

        # final dropout + project
        layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # supports [..., emb_dim] (e.g. [B, L, D] or [N, D])
        x = self.norm(x)
        return self.mlp(x)


def build_model_from_cfg(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build the mlp2 residue model from run.cfg.
    """
    if cfg.get("model_type") != "mlp2":
        raise SystemExit(
            "Only model_type='mlp2' is supported in this script; "
            f"got {cfg.get('model_type')}"
        )

    # MLPHead2 pulls emb_dim, hid_dim, out_dim, num_layers, dropout
    # directly from cfg
    return MLPHead2(cfg)


# ---------------------------------------------------------------------------
#  Shared hard-coded head configs
# ---------------------------------------------------------------------------

# SSBind, LigBind, ZNBind all share this head architecture
SHARED_HEAD_CFG: Dict[str, Any] = {
    "model_type": "mlp2",
    "emb_dim": 2560,
    "hid_dim": 2560,
    "out_dim": 1,
    "num_layers": 1,
    "dropout": 0.1,
}

# LigCys uses its own MLP shape
LIGCYS_HEAD_CFG: Dict[str, Any] = {
    "model_type": "mlp2",
    "emb_dim": 2560,
    "hid_dim": [1024, 516, 256],
    "out_dim": 1,
    "num_layers": 3,
    "dropout": 0.5,
}


# ---------------------------------------------------------------------------
#  Fixed ESM-C + ROI configuration for this deployed model
# ---------------------------------------------------------------------------

# ESM-C backbone configuration (never changed in production)
ESMC_LAYER = 76
ESMC_MODEL_NAME = "esmc-6b-2024-12"

# ROI behavior:
# - SSBind and LigCys: cysteines only
# - LigBind and ZnBind: all residues
ROI_SS_LIGCYS = "C"
ROI_LIG_ZN = "*"


# ---------------------------------------------------------------------------
#  3. Load ensemble checkpoints
# ---------------------------------------------------------------------------

def count_checkpoints(ckpt_dir: str) -> int:
    """
    Count .pt checkpoint files in a directory.
    """
    return sum(
        1 for f in os.listdir(ckpt_dir)
        if f.endswith(".pt")
    )


def load_checkpoints(
    ckpt_dir: str,
    device: torch.device,
    progress=None
) -> List[Dict[str, torch.Tensor]]:
    """
    Load all .pt files from a directory. Each may be either:
      * a full dict containing 'model_state_dict', or
      * a bare state_dict.
    Returns a list of state_dicts.
    """
    fns = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".pt")
    )
    if not fns:
        raise SystemExit(f"No .pt checkpoints found in {ckpt_dir}")

    state_dicts = []
    for fn in fns:
        path = os.path.join(ckpt_dir, fn)
        data = torch.load(path, map_location=device, weights_only=False)
        sd = data.get("model_state_dict", data)
        state_dicts.append(sd)
        if progress is not None:
            progress.update(1)
    return state_dicts


def build_ensemble(
    cfg: Dict[str, Any],
    ckpt_dir: str,
    device: torch.device,
    progress=None
) -> List[nn.Module]:
    """
    Instantiate one model per checkpoint and load its weights.
    """
    sds = load_checkpoints(ckpt_dir, device, progress=progress)
    models: List[nn.Module] = []

    for sd in sds:
        mdl = build_model_from_cfg(cfg)
        mdl.load_state_dict(sd)
        mdl.to(device)
        mdl.eval()
        models.append(mdl)

    return models


# ---------------------------------------------------------------------------
#  4. ESM-C Forge extraction for a single sequence
# ---------------------------------------------------------------------------

def get_forge_token(token_arg: str) -> str:
    """
    If token_arg is a path to a file, read the token from that file.
    Otherwise treat token_arg as the token string itself.
    """
    if os.path.isfile(token_arg):
        with open(token_arg, "r") as f:
            token = f.read().strip()
        return token
    return token_arg.strip()


def retry_operation(
    func,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.5
):
    """
    Retry func() up to max_retries with exponential backoff.
    Copied from plmpg-esmC-extractor.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries:
                sleep_t = delay + random.uniform(0.0, jitter)
                print(
                    "Attempt "
                    f"{attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {sleep_t:.2f}s..."
                )
                time.sleep(sleep_t)
                delay *= backoff_factor
            else:
                print(
                    "Attempt "
                    f"{attempt}/{max_retries} failed: {e}. "
                    "No more retries."
                )
                raise


def extract_layer(
    forge_client: ESM3ForgeInferenceClient,
    protein_tensor: Any,
    layer_i: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Get hidden states for layer_i with retries.
    Copied from plmpg-esmC-extractor.
    """
    def do_extract():
        cfg_l = LogitsConfig(
            return_hidden_states=True,
            ith_hidden_layer=layer_i,
        )
        out = forge_client.logits(protein_tensor, cfg_l)
        if out.hidden_states is None:
            raise ValueError(f"No hidden states for layer {layer_i}")
        return out.hidden_states.squeeze()

    return retry_operation(
        do_extract,
        max_retries=cfg["max_retries"],
        initial_delay=cfg["initial_delay"],
        backoff_factor=cfg["backoff_factor"],
        jitter=cfg["jitter"],
    )


def extract_esmc_layer(
    seq: str,
    layer: int = 76,
    mdl: str = "esmc-6b-2024-12",
    forge_url: str = "https://forge.evolutionaryscale.ai",
    forge_token: str = "",
) -> torch.Tensor:
    """
    Use ESM-C Forge to get hidden states for a single sequence at a
    given layer.

    Returns a Tensor of shape [L+1, D] where index 0 is BOS,
    residues 1..L correspond to seq positions 1..L.
    Wired to match plmpg-esmC-extractor behavior.
    """
    client = ESM3ForgeInferenceClient(
        model=mdl,
        url=forge_url,
        token=forge_token,
    )

    prot = ESMProtein(
        sequence=seq,
        potential_sequence_of_concern=True,
    )
    protein_tensor = retry_operation(
        lambda: client.encode(prot),
        max_retries=5,
        initial_delay=2,
        backoff_factor=2,
        jitter=0.5,
    )

    cfg = {
        "max_retries": 5,
        "initial_delay": 2,
        "backoff_factor": 2,
        "jitter": 0.5,
    }
    hstate = extract_layer(client, protein_tensor, layer, cfg)

    # Expect [L+1, D]
    if hstate.dim() != 2:
        raise RuntimeError(
            "Expected hidden states of shape [L+1, D], got "
            f"{tuple(hstate.shape)}"
        )
    return hstate


# ---------------------------------------------------------------------------
#  5. Run ensemble inference on a sequence
# ---------------------------------------------------------------------------

def ensemble_predict_on_sequence(
    seq_id: str,
    seq: str,
    models: List[nn.Module],
    forge_token: str,
    forge_url: str = "https://forge.evolutionaryscale.ai",
    layer: int = 76,
    mdl_name: str = "esmc-6b-2024-12",
    roi_letters: str = "C",
    agg: str = "max",
    trunto: int = 2046,
    device: torch.device | None = None,
    vote_thr: float = 0.5,
    topk_k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the ensemble on a single sequence.

    Returns:
      positions : array of (1-based) positions of ROI residues in the
                  original seq
      per_model : array of shape [N, M] of per-ROI, per-model
                  probabilities
      agg_probs : array of shape [N] aggregated across models
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    seq = seq.strip().upper()
    L = len(seq)

    # ESM-C hidden states [L+1, D]; index 0 is BOS, 1..L are residues.
    hstate = extract_esmc_layer(
        seq=seq,
        layer=layer,
        mdl=mdl_name,
        forge_url=forge_url,
        forge_token=forge_token,
    ).to(device)

    if hstate.size(0) <= 1:
        raise RuntimeError(
            "Hidden states have only "
            f"{hstate.size(0)} tokens; expected > 1 for sequence of "
            f"length {L}."
        )

    # ROI handling: '*' means all residues; otherwise filter by letters
    max_res = min(trunto, L)
    use_all_residues = roi_letters.strip() == "*"
    roi_set: set[str] | None = None
    if not use_all_residues:
        roi_set = set(roi_letters.upper())

    positions: List[int] = []
    emb_rows: List[torch.Tensor] = []
    for i, aa in enumerate(seq, start=1):
        if i > max_res:
            break

        if use_all_residues or (roi_set is not None and aa in roi_set):
            positions.append(i)
            if i < hstate.size(0):
                emb_rows.append(hstate[i])
            else:
                # Should not happen unless ESM truncates unusually
                raise RuntimeError(
                    "Hidden state length "
                    f"{hstate.size(0)} < residue position {i}"
                )

    if not emb_rows:
        print(
            f"[{seq_id}] No ROI residues ({roi_letters}) found in "
            "sequence."
        )
        return (
            np.array([], dtype=int),
            np.zeros((0, len(models))),
            np.array([]),
        )

    # Build [1, N, D] batch
    emb_tensor = torch.stack(emb_rows, dim=0)   # [N, D]
    emb_tensor = emb_tensor.unsqueeze(0)        # [1, N, D]

    per_model_probs: List[np.ndarray] = []

    with torch.no_grad():
        for net in models:
            target_dtype = next(net.parameters()).dtype
            logits = net(emb_tensor.to(dtype=target_dtype))
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits[:, :, 0]
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            per_model_probs.append(probs)

    per_model_probs_arr = np.stack(per_model_probs, axis=1)
    N, M = per_model_probs_arr.shape

    if agg == "max":
        agg_probs = per_model_probs_arr.max(axis=1)

    elif agg == "min":
        agg_probs = per_model_probs_arr.min(axis=1)

    elif agg == "avg":
        agg_probs = per_model_probs_arr.mean(axis=1)

    elif agg == "vote":
        votes = (per_model_probs_arr >= vote_thr).astype(int)
        agg_probs = votes.mean(axis=1)

    elif agg == "topk":
        # plmpg-style per-model, per-protein Top-K voting:
        # each model votes for its K highest-prob residues; y_score
        # = votes / M
        k = 1 if topk_k is None else int(topk_k)
        k = max(1, min(k, N))

        vote_counts = np.zeros((N,), dtype=np.int32)
        for m in range(M):
            sc = per_model_probs_arr[:, m]
            k_m = min(k, N)
            if k_m == N:
                vote_counts += 1
            else:
                top_idx = np.argpartition(sc, -k_m)[-k_m:]
                vote_counts[top_idx] += 1

        agg_probs = vote_counts.astype(np.float32) / float(M)

    else:
        raise ValueError(f"Unknown aggregation: {agg}")

    return (
        np.array(positions, dtype=int),
        per_model_probs_arr,
        agg_probs,
    )


# ---------------------------------------------------------------------------
#  6. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone SSBind/LigBind/ZNBind/LigCys ESM-C ensemble "
            "inference (no plmpg import; residue mode; fixed ESM "
            "layer/model/ROI/token)."
        )
    )
    p.add_argument(
        "--ssbind",
        help="Directory containing SSBind ensemble checkpoints (.pt).",
    )
    p.add_argument(
        "--ligbind",
        help="Directory containing LigBind ensemble checkpoints (.pt).",
    )
    p.add_argument(
        "--znbind",
        help="Directory containing ZnBind ensemble checkpoints (.pt).",
    )
    p.add_argument(
        "--ligcys",
        action="append",
        help=(
            "Directory containing LigCys ensemble checkpoints (.pt). "
            "Use multiple times for multiple LigCys ensembles."
        ),
    )
    p.add_argument(
        "--ligbindtopk",
        type=int,
        default=None,
        help=(
            "If set and LigBind is used, only use the top N residues "
            "ranked by LigBind probability when deciding which rows "
            "to show."
        ),
    )
    p.add_argument(
        "--sequence",
        help="Amino-acid sequence string to score.",
    )
    p.add_argument(
        "--fasta",
        help=(
            "Optional FASTA file; if provided, all sequences in it are "
            "scored."
        ),
    )
    p.add_argument(
        "--id",
        default="query",
        help="Identifier for --sequence (used in output).",
    )
    p.add_argument(
        "--trunto",
        type=int,
        default=2046,
        help="Max residues to consider (training default for esmc).",
    )
    p.add_argument(
        "--out",
        help=(
            "Write prediction table to this file instead of standard "
            "output."
        ),
    )
    p.add_argument(
        "--forge-token",
        help=(
            "Either the Forge token string, or a path to a file "
            "containing the token. If omitted, a cached token from "
            f"{TOKEN_FILE} (or $AIPP_FORGE_TOKEN_FILE) will be used."
        ),
    )
    p.add_argument(
        "--forge-url",
        default="https://forge.evolutionaryscale.ai",
        help="Forge URL (default: https://forge.evolutionaryscale.ai).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return p.parse_args()


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Very small FASTA reader: returns list of (header, seq).
    Header does not include '>'.
    """
    records = []
    with open(path, "r") as f:
        hdr = None
        chunks: List[str] = []
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                if hdr is not None:
                    records.append((hdr, "".join(chunks)))
                hdr = ln[1:].strip()
                chunks = []
            else:
                chunks.append(ln)
        if hdr is not None:
            records.append((hdr, "".join(chunks)))
    return records


def main():
    args = parse_args()

    cmd_str = " ".join(
        shlex.quote(a) for a in sys.argv
    )
    print("    "+cmd_str+'''\n
     ------------------------------------------------''')

    if not args.sequence and not args.fasta:
        raise SystemExit("You must provide either --sequence or --fasta")

    # ------------------------------------------------------------------
    # Resolve ensemble directories (CLI overrides defaults under wts/)
    # ------------------------------------------------------------------
    ssbind_dir = args.ssbind or os.path.join(
        DEFAULT_WTS_ROOT,
        "ssbind_v1",
    )
    if ssbind_dir and not os.path.isdir(ssbind_dir):
        print(f"Warning: SSBind directory not found: {ssbind_dir}")
        ssbind_dir = None

    ligbind_dir = args.ligbind or os.path.join(
        DEFAULT_WTS_ROOT,
        "ligbind_v1",
    )
    if ligbind_dir and not os.path.isdir(ligbind_dir):
        print(f"Warning: LigBind directory not found: {ligbind_dir}")
        ligbind_dir = None

    znbind_dir = args.znbind or os.path.join(
        DEFAULT_WTS_ROOT,
        "znbind_v1",
    )
    if znbind_dir and not os.path.isdir(znbind_dir):
        print(f"Warning: ZnBind directory not found: {znbind_dir}")
        znbind_dir = None

    # LigCys: either explicit dirs, or known defaults under DEFAULT_WTS_ROOT
    if args.ligcys:
        ligcys_dirs: List[str] | None = []
        for d in args.ligcys:
            if os.path.isdir(d):
                ligcys_dirs.append(d)
            else:
                print(f"Warning: LigCys directory not found: {d}")
        if not ligcys_dirs:
            ligcys_dirs = None
    else:
        ligcys_dirs = []
        for sub in ("ligcysA_v1", "ligcysS_v1"):
            candidate = os.path.join(DEFAULT_WTS_ROOT, sub)
            if os.path.isdir(candidate):
                ligcys_dirs.append(candidate)
        if not ligcys_dirs:
            ligcys_dirs = None

    # At least one task ensemble must be provided or found by default
    if not (ssbind_dir or ligbind_dir or znbind_dir or ligcys_dirs):
        raise SystemExit(
            "No ensemble directories found. Provide at least one of "
            "--ssbind, --ligbind, --znbind, or --ligcys, or ensure "
            "default weights exist under "
            f"{DEFAULT_WTS_ROOT}."
        )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Forge token: CLI (if provided) wins and is cached; otherwise reuse
    # or prompt interactively if none exists
    # ------------------------------------------------------------------
    if args.forge_token:
        forge_token = get_forge_token(args.forge_token)
        try:
            save_token(TOKEN_FILE, forge_token)
            print(f"Saved Forge token to {TOKEN_FILE}")
        except OSError as e:
            print(
                "Warning: could not save Forge token to "
                f"{TOKEN_FILE}: {e}"
            )
    else:
        forge_token = load_saved_token(TOKEN_FILE)
        if forge_token:
            print(f"Using Forge token from {TOKEN_FILE}")
        else:
            print(
                "No Forge token provided on the command line and none "
                f"found in {TOKEN_FILE}."
            )
            try:
                token_in = getpass.getpass(
                    "Please enter a Forge token "
                    "(input hidden, will be cached): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("No Forge token provided; aborting.")

            if not token_in:
                raise SystemExit("Empty Forge token provided; aborting.")

            forge_token = token_in
            try:
                save_token(TOKEN_FILE, forge_token)
                print(f"Saved Forge token to {TOKEN_FILE}")
            except OSError as e:
                print(
                    "Warning: could not save Forge token to "
                    f"{TOKEN_FILE}: {e}"
                )

    # ------------------------------------------------------------------
    # Count checkpoints and create a single init progress bar
    # ------------------------------------------------------------------
    total_ckpts = 0
    for d in (ssbind_dir, ligbind_dir, znbind_dir):
        if d:
            total_ckpts += count_checkpoints(d)
    if ligcys_dirs:
        for d in ligcys_dirs:
            total_ckpts += count_checkpoints(d)

    init_pbar = None
    if total_ckpts > 0:
        init_pbar = tqdm(
            total=total_ckpts,
            desc="initializing AiPP backend",
            unit="ckpt",
            ncols=80,
            leave=False,
        )

    # ------------------------------------------------------------------
    # Build ensembles per task
    # ------------------------------------------------------------------
    ssbind_models: List[nn.Module] | None = None
    ligbind_models: List[nn.Module] | None = None
    znbind_models: List[nn.Module] | None = None
    ligcys_ensembles: List[List[nn.Module]] | None = None

    if ssbind_dir:
        ssbind_models = build_ensemble(
            SHARED_HEAD_CFG,
            ssbind_dir,
            device,
            progress=init_pbar,
        )

    if ligbind_dir:
        ligbind_models = build_ensemble(
            SHARED_HEAD_CFG,
            ligbind_dir,
            device,
            progress=init_pbar,
        )

    if znbind_dir:
        znbind_models = build_ensemble(
            SHARED_HEAD_CFG,
            znbind_dir,
            device,
            progress=init_pbar,
        )

    if ligcys_dirs:
        ligcys_ensembles = []
        for d in ligcys_dirs:
            models = build_ensemble(
                LIGCYS_HEAD_CFG,
                d,
                device,
                progress=init_pbar,
            )
            ligcys_ensembles.append(models)
        if not ligcys_ensembles:
            ligcys_ensembles = None

    if init_pbar is not None:
        init_pbar.close()

    if not (ssbind_models or ligbind_models or znbind_models
            or ligcys_ensembles):
        raise SystemExit(
            "No ensembles built — check the provided directories."
        )

    tasks_order = ["SSBind", "LigBind", "ZNBind", "LigCys"]

    out_fh = None
    if args.out:
        try:
            out_fh = open(args.out, "w", encoding="utf-8")
        except OSError as e:
            raise SystemExit(
                f"Could not open output file {args.out}: {e}"
            )

    def run_one(seq_id: str, seq: str):
        #print(f"\n=== {seq_id} ===")
        seq_u = seq.strip().upper()

        # task_name -> {pos: prob}
        task_pos_probs: Dict[str, Dict[int, float]] = {}

        # SSBind: ROI always 'C', agg=max
        if ssbind_models:
            positions, _, agg_probs = ensemble_predict_on_sequence(
                seq_id=seq_id,
                seq=seq,
                models=ssbind_models,
                forge_token=forge_token,
                forge_url=args.forge_url,
                layer=ESMC_LAYER,
                mdl_name=ESMC_MODEL_NAME,
                roi_letters=ROI_SS_LIGCYS,
                agg="max",
                trunto=args.trunto,
                device=device,
            )
            task_pos_probs["SSBind"] = {
                int(pos): float(agg_probs[i])
                for i, pos in enumerate(positions)
            }

        # LigBind: ROI = all residues, agg=avg
        if ligbind_models:
            positions, _, agg_probs = ensemble_predict_on_sequence(
                seq_id=seq_id,
                seq=seq,
                models=ligbind_models,
                forge_token=forge_token,
                forge_url=args.forge_url,
                layer=ESMC_LAYER,
                mdl_name=ESMC_MODEL_NAME,
                roi_letters=ROI_LIG_ZN,
                agg="avg",
                trunto=args.trunto,
                device=device,
            )
            task_pos_probs["LigBind"] = {
                int(pos): float(agg_probs[i])
                for i, pos in enumerate(positions)
            }

        # ZnBind: ROI = all residues, agg=avg
        if znbind_models:
            positions, _, agg_probs = ensemble_predict_on_sequence(
                seq_id=seq_id,
                seq=seq,
                models=znbind_models,
                forge_token=forge_token,
                forge_url=args.forge_url,
                layer=ESMC_LAYER,
                mdl_name=ESMC_MODEL_NAME,
                roi_letters=ROI_LIG_ZN,
                agg="avg",
                trunto=args.trunto,
                device=device,
            )
            task_pos_probs["ZNBind"] = {
                int(pos): float(agg_probs[i])
                for i, pos in enumerate(positions)
            }

        # LigCys: ROI always 'C'; per-ensemble Top-K=1 voting, then
        # average per-residue scores across ensembles.
        if ligcys_ensembles:
            ligcys_pos_scores: Dict[int, List[float]] = {}
            common_positions: np.ndarray | None = None

            for models in ligcys_ensembles:
                if not models:
                    continue

                positions, _, agg_probs = (
                    ensemble_predict_on_sequence(
                        seq_id=seq_id,
                        seq=seq,
                        models=models,
                        forge_token=forge_token,
                        forge_url=args.forge_url,
                        layer=ESMC_LAYER,
                        mdl_name=ESMC_MODEL_NAME,
                        roi_letters=ROI_SS_LIGCYS,
                        agg="topk",
                        trunto=args.trunto,
                        device=device,
                        topk_k=1,
                    )
                )

                if positions.size == 0:
                    continue

                if common_positions is None:
                    common_positions = positions
                elif not np.array_equal(common_positions, positions):
                    raise RuntimeError(
                        "LigCys ensembles produced different ROI "
                        "positions; cannot average per-residue "
                        "Top-K scores safely."
                    )

                for i, pos in enumerate(positions):
                    ligcys_pos_scores.setdefault(
                        int(pos), []
                    ).append(float(agg_probs[i]))

            if ligcys_pos_scores:
                task_pos_probs["LigCys"] = {
                    pos: (sum(vals) / len(vals))
                    for pos, vals in ligcys_pos_scores.items()
                }

        if not task_pos_probs:
            print("No ROI residues found for any task.")
            return

        # Position sets per task
        pos_ss = set(task_pos_probs.get("SSBind", {}).keys())
        pos_lb = set(task_pos_probs.get("LigBind", {}).keys())
        pos_zn = set(task_pos_probs.get("ZNBind", {}).keys())
        pos_lc = set(task_pos_probs.get("LigCys", {}).keys())

        # LigBind top-N selection (for row inclusion)
        top_lb_positions: set[int] = set()
        if "LigBind" in task_pos_probs:
            lb = task_pos_probs["LigBind"]
            sorted_lb = sorted(
                lb.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if args.ligbindtopk is not None:
                k = max(0, int(args.ligbindtopk))
            else:
                k = 20
            top_lb_positions = {pos for pos, _ in sorted_lb[:k]}

        # Final rows: all SSBind + all LigCys + all ZNBind + LigBind
        # top-N
        row_positions = set()
        row_positions |= pos_ss
        row_positions |= pos_lc
        row_positions |= pos_zn
        row_positions |= top_lb_positions

        if not row_positions:
            print("No positions remain after filtering.")
            return

        # Compute per-task ranks (1 = highest prob)
        task_ranks: Dict[str, Dict[int, int]] = {}
        for t, posprob in task_pos_probs.items():
            items = sorted(
                posprob.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            task_ranks[t] = {
                pos: rank + 1
                for rank, (pos, _) in enumerate(items)
            }

        # Unified table: pos, AA, SSBind, SSBind_topN, LigBind,
        # LigBind_topN, ZNBind, ZNBind_topN, LigCys, LigCys_topN
        dest = out_fh if out_fh is not None else sys.stdout
        header_cols = [
            "pos", "AA",
            "SSBind", "SSBind_topN",
            "LigBind", "LigBind_topN",
            "ZNBind", "ZNBind_topN",
            "LigCys", "LigCys_topN",
        ]
        print("\t".join(header_cols), file=dest)

        for pos in sorted(row_positions):
            aa = (
                seq_u[pos - 1]
                if 1 <= pos <= len(seq_u)
                else "X"
            )
            row: List[str] = [str(pos), aa]
            for t in tasks_order:
                probs = task_pos_probs.get(t)
                ranks = task_ranks.get(t)
                prob = (
                    probs.get(pos)
                    if probs is not None and pos in probs
                    else None
                )
                rank = (
                    ranks.get(pos)
                    if ranks is not None and pos in ranks
                    else None
                )
                row.append(
                    f"{prob:.4f}"
                    if prob is not None
                    else "NA"
                )
                row.append(
                    str(rank)
                    if rank is not None
                    else "NA"
                )
            print("\t".join(row), file=dest)

    # Collect sequences to score
    jobs: List[Tuple[str, str]] = []
    if args.sequence:
        jobs.append((args.id, args.sequence))

    if args.fasta:
        records = read_fasta(args.fasta)
        for hdr, seq in records:
            uid = hdr.split()[0]
            jobs.append((uid, seq))

    if not jobs:
        if out_fh is not None:
            out_fh.close()
        return

    # Inference progress bar over sequences
    inf_pbar = tqdm(
        total=len(jobs),
        desc="running AiPP inference",
        unit="seq",
        ncols=80,
        leave=False,
    )
    try:
        for uid, seq in jobs:
            run_one(uid, seq)
            inf_pbar.update(1)
    finally:
        inf_pbar.close()

    if out_fh is not None:
        out_fh.close()
        print(f"output written to: {args.out}")

    print("")


def splash():
    print(
        """
           .o.        o8o  ooooooooo.   ooooooooo.   
          .888.       `"'  `888   `Y88. `888   `Y88. 
         .8"888.     oooo   888   .d88'  888   .d88' 
        .8' `888.    `888   888ooo88P'   888ooo88P'  
       .88ooo8888.    888   888          888         
      .8'     `888.   888   888          888         
     o88o     o8888o o888o o888o        o888o        


          < command-line inference interface >

            Written by: Guy W. Dayhoff, Ph.D.

     ------------------------------------------------
        """
    )


if __name__ == "__main__":
    splash()
    main()

