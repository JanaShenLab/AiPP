#!/usr/bin/env python3

import argparse
import datetime
import gc
import getpass
import os
import random
import shlex
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Global paths and fixed model settings
# ---------------------------------------------------------------------------
#
# This program extracts one ESM-C residue embedding tensor per sequence and
# then reuses that same tensor across every task. That avoids doing the most
# expensive step over and over.
#
# Model execution is streamed one checkpoint at a time so that GPU memory
# pressure stays low. If CUDA runs out of memory, the script switches itself
# to CPU and finishes the run there instead of just crashing.
#

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_WTS_ROOT = os.environ.get(
    "AIPP_WTS_DIR",
    os.path.join(HERE, "env/wts"),
)

TOKEN_FILE = os.environ.get(
    "AIPP_FORGE_TOKEN_FILE",
    os.path.join(os.path.expanduser("~"), ".aipp_forge_token"),
)

ESMC_LAYER = 76
ESMC_MODEL_NAME = "esmc-6b-2024-12"

ROI_ALL = "all"
ROI_CYS = "cys"

THRESH_SSBIND = 0.5045
THRESH_ZNBIND = 0.6958
THRESH_CUBIND = 0.5164
THRESH_FEBIND = 0.5470
THRESH_FESBIND = 0.5799
THRESH_HEMBIND = 0.6287

LIGCYS_CGRV_BETA = 0.5
LIGCYS_CGRV_RHO = 0.60
LIGCYS_CGRV_S2 = 0.20
LIGCYS_CGRV_S3 = 0.40
LIGCYS_CGRV_USE_SECONDARY = True
LIGCYS_CGRV_USE_TERTIARY = True
LIGCYS_FDR_PERMS = 200
LIGCYS_FDR_SEED = 0


# ---------------------------------------------------------------------------
# Small console helper
# ---------------------------------------------------------------------------

def log_info(msg: str) -> None:
    """
    Print a normal message above tqdm's live progress bar.
    """
    tqdm.write(msg, file=sys.stdout)


# ---------------------------------------------------------------------------
# Model heads
# ---------------------------------------------------------------------------
#
# Each task is just a small classifier head on top of the fixed ESM-C residue
# embedding. Most tasks share the same head shape. LigCys uses a deeper one.
#

class MLPHead(nn.Module):
    """
    Small MLP head used by the residue models.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        emb_dim = cfg["emb_dim"]
        hid_dim = cfg["hid_dim"]
        out_dim = cfg["out_dim"]
        n_layers = cfg["num_layers"]
        dropout_p = cfg["dropout"]

        self.norm = nn.LayerNorm(emb_dim)

        if isinstance(hid_dim, (list, tuple)):
            if len(hid_dim) != n_layers:
                raise ValueError(
                    "Expected hid_dim list of length "
                    f"{n_layers}, got {len(hid_dim)}"
                )
            hid_dims = list(hid_dim)
        else:
            hid_dims = [hid_dim] * n_layers

        layers: List[nn.Module] = []
        in_dim = emb_dim

        for h_dim in hid_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.mlp(x)


def build_model_from_cfg(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build one model head from a small config dict.
    """
    if cfg.get("model_type") != "mlp":
        raise SystemExit(
            "Only model_type='mlp' is supported in this script; "
            f"got {cfg.get('model_type')}"
        )

    return MLPHead(cfg)


SHARED_HEAD_CFG: Dict[str, Any] = {
    "model_type": "mlp",
    "emb_dim": 2560,
    "hid_dim": 2560,
    "out_dim": 1,
    "num_layers": 1,
    "dropout": 0.1,
}

LIGCYS_HEAD_CFG: Dict[str, Any] = {
    "model_type": "mlp",
    "emb_dim": 2560,
    "hid_dim": [1024, 516, 256],
    "out_dim": 1,
    "num_layers": 3,
    "dropout": 0.5,
}


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
#
# Each task is described once here so that the CLI, defaults, execution
# order, and output formatting stay in sync.
#

@dataclass(frozen=True)
class TaskSpec:
    """
    One task family and how it should be run.
    """

    name: str
    cli_attr: str
    default_subdirs: Tuple[str, ...]
    head_cfg: Dict[str, Any]
    roi_mode: str
    agg: str
    multi_dir: bool = False


TASK_SPECS: Tuple[TaskSpec, ...] = (
    TaskSpec(
        name="SSBind",
        cli_attr="ssbind",
        default_subdirs=("ssbind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="LigBind",
        cli_attr="ligbind",
        default_subdirs=("ligbind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_ALL,
        agg="avg",
    ),
    TaskSpec(
        name="ZnBind",
        cli_attr="znbind",
        default_subdirs=("znbind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="CuBind",
        cli_attr="cubind",
        default_subdirs=("cubind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="FeBind",
        cli_attr="febind",
        default_subdirs=("febind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="FeSBind",
        cli_attr="fesbind",
        default_subdirs=("fesbind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="HemeBind",
        cli_attr="hembind",
        default_subdirs=("hembind_v1",),
        head_cfg=SHARED_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="avg",
    ),
    TaskSpec(
        name="LigCys",
        cli_attr="ligcys",
        default_subdirs=("ligcysA_v1", "ligcysS_v1"),
        head_cfg=LIGCYS_HEAD_CFG,
        roi_mode=ROI_CYS,
        agg="cgrv",
        multi_dir=True,
    ),
)

OUTPUT_TASK_ORDER: Tuple[str, ...] = (
    "LigCys",
    "SSBind",
    "ZnBind",
    "CuBind",
    "FeBind",
    "FeSBind",
    "HemeBind",
    "LigBind",
)

TOPN_TASKS: Tuple[str, ...] = (
    "LigCys",
    "LigBind",
)

AUX_SCORE_BINARY_TASKS: Tuple[str, ...] = (
    "SSBind",
    "ZnBind",
    "CuBind",
    "FeBind",
    "FeSBind",
    "HemeBind",
)

BINARY_THRESHOLDS: Dict[str, float] = {
    "SSBind": THRESH_SSBIND,
    "ZnBind": THRESH_ZNBIND,
    "CuBind": THRESH_CUBIND,
    "FeBind": THRESH_FEBIND,
    "FeSBind": THRESH_FESBIND,
    "HemeBind": THRESH_HEMBIND,
}

ALL_RES_TASKS: Tuple[str, ...] = (
    "LigBind",
)

CYS_TASKS: Tuple[str, ...] = (
    "SSBind",
    "ZnBind",
    "CuBind",
    "FeBind",
    "FeSBind",
    "HemeBind",
    "LigCys",
)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

@dataclass
class RuntimeState:
    """
    Track which device we are using right now.
    """

    device: torch.device
    gpu_fallback_used: bool = False


@dataclass
class TaskRunResult:
    """
    Per-task outputs collected for one sequence.
    """

    scores: Dict[int, float]
    confidence: Optional[Dict[int, float]] = None


# ---------------------------------------------------------------------------
# Token and file helpers
# ---------------------------------------------------------------------------

def load_saved_token(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f_in:
            token = f_in.read().strip()
        return token or None
    except FileNotFoundError:
        return None


def save_token(path: str, token: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f_out:
        f_out.write(token.strip() + "\n")


def get_forge_token(token_arg: str) -> str:
    """
    Accept either a literal token string or a path to a token file.
    """
    if os.path.isfile(token_arg):
        with open(token_arg, "r", encoding="utf-8") as f_in:
            return f_in.read().strip()

    return token_arg.strip()


def retry_operation(
    func,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.5,
):
    """
    Small retry loop for Forge/API calls.
    """
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= max_retries:
                log_info(
                    f"Attempt {attempt}/{max_retries} failed: {exc}. "
                    "No more retries."
                )
                raise

            sleep_t = delay + random.uniform(0.0, jitter)
            log_info(
                f"Attempt {attempt}/{max_retries} failed: {exc}. "
                f"Retrying in {sleep_t:.2f}s..."
            )
            time.sleep(sleep_t)
            delay *= backoff_factor


def clear_cuda_state() -> None:
    """
    Ask PyTorch to release CUDA cache that is no longer needed.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        torch.cuda.empty_cache()

    gc.collect()


def is_cuda_oom(exc: BaseException) -> bool:
    """
    Detect the common CUDA out-of-memory error shapes.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    msg = str(exc).lower()

    if "out of memory" in msg and "cuda" in msg:
        return True

    if "cuda error" in msg and "out of memory" in msg:
        return True

    return False


def list_checkpoint_files(ckpt_dir: str) -> List[str]:
    """
    Return all checkpoint files in a stable order.
    """
    return sorted(
        os.path.join(ckpt_dir, fn)
        for fn in os.listdir(ckpt_dir)
        if fn.endswith(".pt")
    )


def count_checkpoints(ckpt_dir: str) -> int:
    return len(list_checkpoint_files(ckpt_dir))


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Tiny FASTA reader.
    """
    records: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f_in:
        hdr = None
        chunks: List[str] = []

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if hdr is not None:
                    records.append((hdr, "".join(chunks)))
                hdr = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

        if hdr is not None:
            records.append((hdr, "".join(chunks)))

    return records


def splash() -> None:
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
        """,
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# ESM-C extraction
# ---------------------------------------------------------------------------
#
# Forge returns one hidden-state row per token. Row 0 is BOS. The usable
# residue rows start at row 1. We only do this extraction once per sequence.
#

def extract_layer(
    forge_client: ESM3ForgeInferenceClient,
    protein_tensor: Any,
    layer_i: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Fetch one hidden layer from Forge with retries.
    """

    def do_extract():
        logits_cfg = LogitsConfig(
            return_hidden_states=True,
            ith_hidden_layer=layer_i,
        )
        out = forge_client.logits(protein_tensor, logits_cfg)
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
    layer: int = ESMC_LAYER,
    mdl: str = ESMC_MODEL_NAME,
    forge_url: str = "https://forge.evolutionaryscale.ai",
    forge_token: str = "",
) -> torch.Tensor:
    """
    Get one per-token hidden-state tensor for one sequence.

    Returns:
      [L + 1, D] where row 0 is BOS and rows 1..L are residues.
    """
    client = ESM3ForgeInferenceClient(
        model=mdl,
        url=forge_url,
        token=forge_token,
    )

    prot = ESMProtein(
        sequence=seq,
        potential_sequence_of_concern=False,
    )

    protein_tensor = retry_operation(
        lambda: client.encode(prot),
        max_retries=5,
        initial_delay=2.0,
        backoff_factor=2.0,
        jitter=0.5,
    )

    retry_cfg = {
        "max_retries": 5,
        "initial_delay": 2.0,
        "backoff_factor": 2.0,
        "jitter": 0.5,
    }

    hstate = extract_layer(client, protein_tensor, layer, retry_cfg)

    if hstate.dim() != 2:
        raise RuntimeError(
            "Expected hidden states of shape [L+1, D], got "
            f"{tuple(hstate.shape)}"
        )

    return hstate


def prepare_all_residue_view(
    seq: str,
    forge_token: str,
    forge_url: str,
    trunto: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extract once, then keep one CPU tensor for all usable residues.
    """
    seq_u = seq.strip().upper()
    seq_len = len(seq_u)
    max_res = min(trunto, seq_len)

    hstate = extract_esmc_layer(
        seq=seq_u,
        layer=ESMC_LAYER,
        mdl=ESMC_MODEL_NAME,
        forge_url=forge_url,
        forge_token=forge_token,
    ).detach().cpu().float()

    usable = hstate.size(0) - 1
    if usable <= 0:
        raise RuntimeError(
            "Hidden states contain no residue rows after BOS."
        )

    if usable < max_res:
        raise RuntimeError(
            f"Hidden state length {usable} is shorter than the "
            f"requested residue count {max_res}."
        )

    all_emb_cpu = hstate[1:max_res + 1].contiguous()
    all_positions = np.arange(1, max_res + 1, dtype=int)

    return all_emb_cpu, all_positions


def prepare_cys_view(
    seq: str,
    all_emb_cpu: torch.Tensor,
    all_positions: np.ndarray,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Build the smaller cysteine-only tensor from the shared residue tensor.
    """
    seq_u = seq.strip().upper()

    mask_list = [
        seq_u[pos - 1] == "C"
        for pos in all_positions.tolist()
    ]

    if not mask_list or not any(mask_list):
        emb_dim = all_emb_cpu.size(1)
        empty = torch.empty(
            (0, emb_dim),
            dtype=all_emb_cpu.dtype,
        )
        return empty, np.array([], dtype=int)

    mask_t = torch.tensor(mask_list, dtype=torch.bool)
    cys_emb_cpu = all_emb_cpu[mask_t].contiguous()
    cys_positions = all_positions[np.array(mask_list, dtype=bool)]

    return cys_emb_cpu, cys_positions


# ---------------------------------------------------------------------------
# LigCys CGRV + FDR helpers
# ---------------------------------------------------------------------------
#
# LigCys now mirrors the plmpg "cgrv + fdr-no-revoke + confidence=1-q" path.
#
# Two details matter here:
#   1. The score is the reported CGRV vote fraction.
#   2. With fdr-no-revoke, q-values are still computed, but they do not change
#      the score or the positive/negative call. They only feed confidence.
#

@dataclass(frozen=True)
class CGRVParams:
    """
    Parameters controlling CGRV behavior.
    """

    beta: float = 0.5
    rho: float = 0.60
    s2: float = 0.20
    s3: float = 0.40
    use_secondary: bool = True
    use_tertiary: bool = True


@dataclass
class CGRVResult:
    """
    Per-protein CGRV result.
    """

    votes_total: np.ndarray
    votes_primary: np.ndarray
    votes_primary_raw: np.ndarray
    votes_secondary: np.ndarray
    votes_tertiary: np.ndarray

    primary_winner: Optional[int]
    secondary_winner: Optional[int]
    tertiary_winner: Optional[int]

    triggered_reassign: bool
    primary_share: float

    r2: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def _safe_r2_linear_fit(
    y: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fit y ~= a + b*x and return (a, b, R^2).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    if y.size < 2:
        return 0.0, 0.0, 0.0

    x_var = float(np.var(x))
    if x_var == 0.0:
        a_val = float(np.mean(y))
        return a_val, 0.0, 0.0

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
    b_val = cov_xy / x_var
    a_val = y_mean - b_val * x_mean

    y_hat = a_val + b_val * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))

    r2_val = 0.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(a_val), float(b_val), float(r2_val)


def _rank_indices_desc(probs: np.ndarray) -> np.ndarray:
    """
    Return residue indices sorted by probability descending per model.
    """
    return np.argsort(probs, axis=0)[::-1, :]


def _winner_index(votes: np.ndarray) -> Optional[int]:
    """
    Return argmax vote index, or None if all votes are zero.
    """
    if votes.size == 0 or int(votes.max()) == 0:
        return None
    return int(np.argmax(votes))


def cgrv_votes_for_protein(
    probs: np.ndarray,
    params: CGRVParams = CGRVParams(),
) -> CGRVResult:
    """
    Run CGRV on one protein.

    probs has shape (n_residues, n_models).
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be 2D (n_residues, n_models)")

    n_res, m_models = probs.shape
    if n_res == 0 or m_models == 0:
        z = np.zeros((n_res,), dtype=int)
        return CGRVResult(
            votes_total=z.copy(),
            votes_primary=z.copy(),
            votes_primary_raw=z.copy(),
            votes_secondary=z.copy(),
            votes_tertiary=z.copy(),
            primary_winner=None,
            secondary_winner=None,
            tertiary_winner=None,
            triggered_reassign=False,
            primary_share=0.0,
            r2=np.zeros((n_res,), dtype=float),
            f2=np.zeros((n_res,), dtype=float),
            f3=np.zeros((n_res,), dtype=float),
        )

    top1 = np.argmax(probs, axis=0)
    votes_primary = np.bincount(top1, minlength=n_res).astype(int)
    votes_primary_raw = votes_primary.copy()

    primary_winner = int(np.argmax(votes_primary))
    primary_share = float(votes_primary[primary_winner] / m_models)

    triggered_reassign = False
    if primary_share >= params.beta:
        votes_primary[:] = 0
        votes_primary[primary_winner] = m_models
        triggered_reassign = True

    g_val = np.max(probs, axis=0)

    a_vec = np.zeros((n_res,), dtype=float)
    b_vec = np.zeros((n_res,), dtype=float)
    r2 = np.zeros((n_res,), dtype=float)

    for ridx in range(n_res):
        a_fit, b_fit, r2_fit = _safe_r2_linear_fit(
            probs[ridx, :],
            g_val,
        )
        a_vec[ridx] = a_fit
        b_vec[ridx] = b_fit
        r2[ridx] = r2_fit

    resid = probs - (a_vec[:, None] + b_vec[:, None] * g_val[None, :])

    order = _rank_indices_desc(probs)
    f2 = np.zeros((n_res,), dtype=float)
    f3 = np.zeros((n_res,), dtype=float)

    if n_res >= 2:
        top2 = order[1, :]
        f2 = np.bincount(top2, minlength=n_res) / float(m_models)

    if n_res >= 3:
        top3 = order[2, :]
        f3 = np.bincount(top3, minlength=n_res) / float(m_models)

    votes_secondary = np.zeros((n_res,), dtype=int)
    secondary_winner = None

    if params.use_secondary and n_res >= 2:
        eligible2 = (
            (np.arange(n_res) != primary_winner) &
            (f2 >= params.s2) &
            (r2 <= params.rho)
        )
        elig_idx2 = np.where(eligible2)[0]

        if elig_idx2.size > 0:
            for midx in range(m_models):
                best_val = 0.0
                best_r = None

                for ridx in elig_idx2:
                    val = float(resid[ridx, midx])
                    if val > best_val:
                        best_val = val
                        best_r = int(ridx)

                if best_r is not None and best_val > 0.0:
                    votes_secondary[best_r] += 1

            secondary_winner = _winner_index(votes_secondary)

    votes_tertiary = np.zeros((n_res,), dtype=int)
    tertiary_winner = None

    if params.use_tertiary and n_res >= 3:
        exclude = np.zeros((n_res,), dtype=bool)
        exclude[primary_winner] = True
        if secondary_winner is not None:
            exclude[secondary_winner] = True

        eligible3 = (
            (~exclude) &
            (f3 >= params.s3) &
            (r2 <= params.rho)
        )
        elig_idx3 = np.where(eligible3)[0]

        if elig_idx3.size > 0:
            for midx in range(m_models):
                best_val = 0.0
                best_r = None

                for ridx in elig_idx3:
                    val = float(resid[ridx, midx])
                    if val > best_val:
                        best_val = val
                        best_r = int(ridx)

                if best_r is not None and best_val > 0.0:
                    votes_tertiary[best_r] += 1

            tertiary_winner = _winner_index(votes_tertiary)

    votes_total = votes_primary + votes_secondary + votes_tertiary

    return CGRVResult(
        votes_total=votes_total,
        votes_primary=votes_primary,
        votes_primary_raw=votes_primary_raw,
        votes_secondary=votes_secondary,
        votes_tertiary=votes_tertiary,
        primary_winner=primary_winner,
        secondary_winner=secondary_winner,
        tertiary_winner=tertiary_winner,
        triggered_reassign=triggered_reassign,
        primary_share=primary_share,
        r2=r2,
        f2=f2,
        f3=f3,
    )


def cgrv_reporting_votes(res: CGRVResult) -> np.ndarray:
    """
    Match the plmpg reporting semantics for CGRV score output.

    If the dominant-winner rule fired, dissenting primary votes are revoked
    rather than reassigned in the reported score.
    """
    primary_rep = res.votes_primary_raw.copy()

    if res.triggered_reassign and res.primary_winner is not None:
        primary_rep[:] = 0
        primary_rep[res.primary_winner] = res.votes_primary_raw[
            res.primary_winner
        ]

    return primary_rep + res.votes_secondary + res.votes_tertiary


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg q-values for one protein.
    """
    pvals = np.asarray(pvals, dtype=float)
    n_vals = int(pvals.size)
    if n_vals == 0:
        return pvals

    order = np.argsort(pvals)
    ps = pvals[order]
    qs = ps * (
        n_vals / np.arange(1, n_vals + 1, dtype=float)
    )
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    qs = np.clip(qs, 0.0, 1.0)

    out = np.empty_like(qs)
    out[order] = qs
    return out


def _permute_columns(
    arr: np.ndarray,
    rng: np.random.Generator,
    out: np.ndarray,
) -> None:
    """
    out[:, m] = arr[perm_m, m] with one independent permutation per model.
    """
    _n_res, n_models = arr.shape
    for midx in range(n_models):
        out[:, midx] = arr[rng.permutation(arr.shape[0]), midx]


def _stat_counts_cgrv(
    arr: np.ndarray,
    params: CGRVParams,
) -> np.ndarray:
    """
    Per-residue reported CGRV vote counts for one protein.
    """
    res = cgrv_votes_for_protein(arr, params=params)
    return cgrv_reporting_votes(res).astype(int)


def fdr_qvalues_for_protein_cgrv(
    arr: np.ndarray,
    obs_counts: np.ndarray,
    params: CGRVParams,
    n_perms: int,
    seed: int,
    show_progress: bool = False,
    desc: Optional[str] = None,
) -> np.ndarray:
    """
    Empirical per-protein q-values for the CGRV statistic.

    Null model:
      independently permute residue rows within each model column.
    """
    arr = np.asarray(arr, dtype=float)
    obs = np.asarray(obs_counts, dtype=int)

    n_res, n_models = arr.shape
    if n_res == 0:
        return np.zeros((0,), dtype=float)

    if int(n_perms) <= 0:
        return np.ones((n_res,), dtype=float)

    rng = np.random.default_rng(int(seed))
    arr_perm = np.empty_like(arr)

    max_stat = 3 * n_models
    hist = np.zeros((max_stat + 1,), dtype=np.int64)

    perm_iter = range(int(n_perms))
    if show_progress:
        label = "LigCys FDR perms"
        if desc is not None:
            label += f" {desc}"
        perm_iter = tqdm(
            perm_iter,
            desc=label,
            ncols=80,
            leave=False,
            file=sys.stdout,
        )

    for _ in perm_iter:
        _permute_columns(arr, rng, arr_perm)
        s_perm = _stat_counts_cgrv(arr_perm, params)
        s_perm = np.clip(s_perm.astype(int), 0, max_stat)
        hist += np.bincount(s_perm, minlength=max_stat + 1)

    total = int(n_perms) * int(n_res)
    tail = np.cumsum(hist[::-1])[::-1]

    obs_clip = np.clip(obs.astype(int), 0, max_stat)
    pvals = (1.0 + tail[obs_clip]) / (1.0 + float(total))
    return _bh_qvalues(pvals)


def ligcys_confidence_fdr_pos_1minusq(
    q_value: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Mirror the plmpg confidence method 'fdr_pos_1minusq'.

    Positive rows get 1 - q. Negative rows get 0.
    """
    q_value = np.asarray(q_value, dtype=float)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    conf = np.zeros((q_value.shape[0],), dtype=float)
    pos_mask = y_pred.astype(bool)

    q_clipped = np.where(np.isfinite(q_value), q_value, 1.0)
    q_clipped = np.clip(q_clipped, 0.0, 1.0)

    conf[pos_mask] = 1.0 - q_clipped[pos_mask]
    return np.clip(conf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Streamed checkpoint inference
# ---------------------------------------------------------------------------
#
# Normal tasks can be aggregated on the fly and never need a full per-model
# score matrix.
#
# LigCys is different now. CGRV and q-value estimation work on the full
# residue-by-model matrix, so for LigCys we collect those columns while still
# loading one checkpoint at a time.
#

def init_agg_state(
    agg: str,
    n_rows: int,
) -> Dict[str, Any]:
    """
    Create the running state for a simple aggregation mode.
    """
    if agg == "avg":
        return {
            "sum": np.zeros((n_rows,), dtype=np.float64),
            "count": 0,
        }

    if agg == "max":
        return {
            "best": np.full((n_rows,), -np.inf, dtype=np.float32),
            "count": 0,
        }

    if agg == "min":
        return {
            "best": np.full((n_rows,), np.inf, dtype=np.float32),
            "count": 0,
        }

    if agg == "vote":
        return {
            "votes": np.zeros((n_rows,), dtype=np.int32),
            "count": 0,
        }

    raise ValueError(f"Unknown aggregation: {agg}")


def update_agg_state(
    state: Dict[str, Any],
    probs: np.ndarray,
    agg: str,
    vote_thr: float,
) -> None:
    """
    Fold one model's residue scores into the running aggregate.
    """
    if agg == "avg":
        state["sum"] += probs
        state["count"] += 1
        return

    if agg == "max":
        state["best"] = np.maximum(state["best"], probs)
        state["count"] += 1
        return

    if agg == "min":
        state["best"] = np.minimum(state["best"], probs)
        state["count"] += 1
        return

    if agg == "vote":
        state["votes"] += (probs >= vote_thr).astype(np.int32)
        state["count"] += 1
        return

    raise ValueError(f"Unknown aggregation: {agg}")


def finalize_agg_state(
    state: Dict[str, Any],
    agg: str,
) -> np.ndarray:
    """
    Convert the running aggregate into one final score per residue.
    """
    count = int(state["count"])
    if count <= 0:
        raise RuntimeError("No models were aggregated.")

    if agg == "avg":
        return (state["sum"] / float(count)).astype(np.float32)

    if agg in ("max", "min"):
        return state["best"].astype(np.float32)

    if agg == "vote":
        return state["votes"].astype(np.float32) / float(count)

    raise ValueError(f"Unknown aggregation: {agg}")


def probs_for_checkpoint(
    ckpt_path: str,
    head_cfg: Dict[str, Any],
    emb_cpu: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run one checkpoint on one residue matrix and return one score per row.

    Dtype behavior is aligned to plmpg.ensemble:
      - load checkpoint tensors onto the target device
      - move model to target device
      - move input to target device with NO explicit dtype cast
      - return raw numpy probabilities with NO forced float32 cast
    """
    model = None
    payload = None
    state_dict = None
    x_dev = None
    logits = None
    probs = None

    try:
        payload = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = payload.get("model_state_dict", payload)

        model = build_model_from_cfg(head_cfg)
#        print("after build:", next(model.parameters()).dtype)

        model.to(device)
#        print("after to(device):", next(model.parameters()).dtype)

        model.load_state_dict(state_dict)
#        print("after load_state_dict:", next(model.parameters()).dtype)

        model.to(device).eval()

        x_dev = emb_cpu.unsqueeze(0).to(device=device)

        with torch.no_grad():
            logits = model(x_dev)
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits[:, :, 0]

            probs = (
                torch.sigmoid(logits)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        return probs
    finally:
        del probs
        del logits
        del x_dev
        del model
        del state_dict
        del payload

        if device.type == "cuda":
            clear_cuda_state()

def stream_standard_task_once(
    ckpt_dir: str,
    head_cfg: Dict[str, Any],
    emb_cpu: torch.Tensor,
    device: torch.device,
    agg: str,
    vote_thr: float = 0.5,
) -> np.ndarray:
    """
    Run one normal ensemble directory using one checkpoint at a time.
    """
    ckpt_paths = list_checkpoint_files(ckpt_dir)
    if not ckpt_paths:
        raise SystemExit(f"No .pt checkpoints found in {ckpt_dir}")

    n_rows = emb_cpu.size(0)
    if n_rows <= 0:
        return np.array([], dtype=np.float32)

    state = init_agg_state(agg, n_rows)

    for ckpt_path in ckpt_paths:
        probs = probs_for_checkpoint(
            ckpt_path=ckpt_path,
            head_cfg=head_cfg,
            emb_cpu=emb_cpu,
            device=device,
        )
        update_agg_state(
            state=state,
            probs=probs,
            agg=agg,
            vote_thr=vote_thr,
        )

    return finalize_agg_state(state, agg)


def collect_prob_matrix_from_dirs(
    task_dirs: Sequence[str],
    head_cfg: Dict[str, Any],
    emb_cpu: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Collect the full residue-by-model probability matrix across all supplied
    checkpoint directories.

    For LigCys this means every checkpoint column contributes directly to the
    CGRV vote calculation.
    """
    n_rows = emb_cpu.size(0)
    if n_rows <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    cols: List[np.ndarray] = []

    for ckpt_dir in task_dirs:
        ckpt_paths = list_checkpoint_files(ckpt_dir)
        if not ckpt_paths:
            raise SystemExit(f"No .pt checkpoints found in {ckpt_dir}")

        for ckpt_path in ckpt_paths:
            probs = probs_for_checkpoint(
                ckpt_path=ckpt_path,
                head_cfg=head_cfg,
                emb_cpu=emb_cpu,
                device=device,
            )
            cols.append(probs)

    if not cols:
        return np.zeros((n_rows, 0), dtype=float)

    return np.stack(cols, axis=1)

def run_standard_task_once(
    spec: TaskSpec,
    task_dirs: List[str],
    emb_cpu: torch.Tensor,
    positions: np.ndarray,
    device: torch.device,
) -> TaskRunResult:
    """
    Run one non-LigCys task on one sequence on one concrete device.
    """
    if positions.size == 0 or emb_cpu.size(0) == 0:
        return TaskRunResult(scores={})

    if spec.multi_dir:
        per_dir_scores: List[np.ndarray] = []

        for ckpt_dir in task_dirs:
            agg_probs = stream_standard_task_once(
                ckpt_dir=ckpt_dir,
                head_cfg=spec.head_cfg,
                emb_cpu=emb_cpu,
                device=device,
                agg=spec.agg,
            )
            per_dir_scores.append(agg_probs)

        if not per_dir_scores:
            return TaskRunResult(scores={})

        mean_scores = np.mean(
            np.stack(per_dir_scores, axis=0),
            axis=0,
        ).astype(np.float32)

        return TaskRunResult(
            scores={
                int(pos): float(mean_scores[i])
                for i, pos in enumerate(positions)
            }
        )

    if len(task_dirs) != 1:
        raise RuntimeError(
            f"{spec.name} expected exactly one directory, got "
            f"{len(task_dirs)}"
        )

    agg_probs = stream_standard_task_once(
        ckpt_dir=task_dirs[0],
        head_cfg=spec.head_cfg,
        emb_cpu=emb_cpu,
        device=device,
        agg=spec.agg,
    )

    return TaskRunResult(
        scores={
            int(pos): float(agg_probs[i])
            for i, pos in enumerate(positions)
        }
    )


def run_ligcys_once(
    spec: TaskSpec,
    task_dirs: List[str],
    emb_cpu: torch.Tensor,
    positions: np.ndarray,
    device: torch.device,
    seq_id: str,
) -> TaskRunResult:
    """
    Run LigCys using:
      pooled checkpoint columns -> CGRV -> FDR q-values ->
      confidence = 1 - q for predicted positives.

    All supplied LigCys checkpoints are pooled into one model matrix.
    """
    if positions.size == 0 or emb_cpu.size(0) == 0:
        return TaskRunResult(scores={}, confidence={})

    probs_mat = collect_prob_matrix_from_dirs(
        task_dirs=task_dirs,
        head_cfg=spec.head_cfg,
        emb_cpu=emb_cpu,
        device=device,
    )
    n_rows, n_models = probs_mat.shape
    if n_rows == 0 or n_models == 0:
        return TaskRunResult(scores={}, confidence={})

    params = CGRVParams(
        beta=LIGCYS_CGRV_BETA,
        rho=LIGCYS_CGRV_RHO,
        s2=LIGCYS_CGRV_S2,
        s3=LIGCYS_CGRV_S3,
        use_secondary=LIGCYS_CGRV_USE_SECONDARY,
        use_tertiary=LIGCYS_CGRV_USE_TERTIARY,
    )

    print("oneoff positions:", positions.tolist())
    print("oneoff top1:", np.argmax(probs_mat, axis=0).tolist())
    cgrv_res = cgrv_votes_for_protein(probs_mat, params=params)
    print("oneoff primary_raw:", cgrv_res.votes_primary_raw.tolist())
    print("oneoff secondary:", cgrv_res.votes_secondary.tolist())
    print("oneoff tertiary:", cgrv_res.votes_tertiary.tolist())


    log_info(
        f"[{seq_id}] LigCys: running CGRV across {n_models} model "
        "column(s)..."
    )
    cgrv_res = cgrv_votes_for_protein(probs_mat, params=params)
    vote_counts = cgrv_reporting_votes(cgrv_res).astype(int)
    y_score = vote_counts / n_models

    log_info(
        f"[{seq_id}] LigCys: computing per-protein FDR q-values "
        f"({LIGCYS_FDR_PERMS} permutations)..."
    )
    q_value = fdr_qvalues_for_protein_cgrv(
        arr=probs_mat,
        obs_counts=vote_counts,
        params=params,
        n_perms=LIGCYS_FDR_PERMS,
        seed=LIGCYS_FDR_SEED ^ int(
            zlib.adler32(seq_id.encode("utf-8")) & 0xffffffff
        ),
        show_progress=True,
        desc=seq_id,
    )

    # Match the plmpg --cgrv --fdr --fdr-no-revoke behavior:
    # q-values are computed but do not revoke or zero scores.
    thr_eff = np.nextafter(0.0, 1.0)
    y_pred = (y_score >= thr_eff).astype(np.int8)

    confidence = ligcys_confidence_fdr_pos_1minusq(
        q_value=q_value,
        y_pred=y_pred,
    )

    return TaskRunResult(
        scores={
            int(pos): float(y_score[i])
            for i, pos in enumerate(positions)
        },
        confidence={
            int(pos): float(confidence[i])
            for i, pos in enumerate(positions)
        },
    )


def run_task_once(
    spec: TaskSpec,
    task_dirs: List[str],
    emb_cpu: torch.Tensor,
    positions: np.ndarray,
    device: torch.device,
    seq_id: str,
) -> TaskRunResult:
    """
    Dispatch one task to the right implementation.
    """
    if spec.name == "LigCys":
        return run_ligcys_once(
            spec=spec,
            task_dirs=task_dirs,
            emb_cpu=emb_cpu,
            positions=positions,
            device=device,
            seq_id=seq_id,
        )

    return run_standard_task_once(
        spec=spec,
        task_dirs=task_dirs,
        emb_cpu=emb_cpu,
        positions=positions,
        device=device,
    )


def run_task_with_fallback(
    spec: TaskSpec,
    seq_id: str,
    task_dirs: List[str],
    emb_cpu: torch.Tensor,
    positions: np.ndarray,
    runtime: RuntimeState,
) -> TaskRunResult:
    """
    Try the active device first. If CUDA runs out of VRAM, switch the rest
    of the run to CPU and rerun the same task there.
    """
    while True:
        try:
            return run_task_once(
                spec=spec,
                task_dirs=task_dirs,
                emb_cpu=emb_cpu,
                positions=positions,
                device=runtime.device,
                seq_id=seq_id,
            )
        except Exception as exc:
            use_fallback = (
                runtime.device.type == "cuda" and is_cuda_oom(exc)
            )
            err_msg = str(exc)
            del exc

            if not use_fallback:
                raise

            log_info(
                f"[{seq_id}] CUDA ran out of VRAM while running "
                f"{spec.name}. Switching to CPU for the rest of "
                "this run."
            )
            log_info(f"[{seq_id}] CUDA error: {err_msg}")

            runtime.device = torch.device("cpu")
            runtime.gpu_fallback_used = True
            clear_cuda_state()


# ---------------------------------------------------------------------------
# CLI args and directory resolution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone SSBind/LigBind/ZnBind/CuBind/FeBind/FeSBind/"
            "HemeBind/LigCys ESM-C ensemble inference."
        )
    )

    p.add_argument(
        "--ssbind",
        help="Directory containing SSBind checkpoints (.pt).",
    )
    p.add_argument(
        "--ligbind",
        help="Directory containing LigBind checkpoints (.pt).",
    )
    p.add_argument(
        "--znbind",
        help="Directory containing ZnBind checkpoints (.pt).",
    )
    p.add_argument(
        "--cubind",
        help="Directory containing CuBind checkpoints (.pt).",
    )
    p.add_argument(
        "--febind",
        help="Directory containing FeBind checkpoints (.pt).",
    )
    p.add_argument(
        "--fesbind",
        help="Directory containing FeSBind checkpoints (.pt).",
    )
    p.add_argument(
        "--hembind",
        help="Directory containing HemeBind checkpoints (.pt).",
    )
    p.add_argument(
        "--ligcys",
        action="append",
        help=(
            "Directory containing LigCys checkpoints (.pt). "
            "Use multiple times for multiple LigCys checkpoint dirs."
        ),
    )
    p.add_argument(
        "--ligbindtopk",
        type=int,
        default=None,
        help=(
            "If set and LigBind is used, only show the top N LigBind "
            "residues in the final table. Default is 20."
        ),
    )
    p.add_argument(
        "--sequence",
        help="Amino-acid sequence string to score.",
    )
    p.add_argument(
        "--fasta",
        help="Optional FASTA file; if set, score all sequences in it.",
    )
    p.add_argument(
        "--id",
        default="query",
        help="Identifier for --sequence output.",
    )
    p.add_argument(
        "--trunto",
        type=int,
        default=2046,
        help="Max residues to consider.",
    )
    p.add_argument(
        "--out",
        help="Write output here instead of stdout.",
    )
    p.add_argument(
        "--forge-token",
        help=(
            "Forge token string, or a path to a file containing it. "
            f"If omitted, a cached token from {TOKEN_FILE} is used."
        ),
    )
    p.add_argument(
        "--forge-url",
        default="https://forge.evolutionaryscale.ai",
        help="Forge URL.",
    )
    p.add_argument(
        "--cpu",
        dest="force_cpu",
        action="store_true",
        help="Force CPU.",
    )
    p.add_argument(
        "--nogpu",
        dest="force_cpu",
        action="store_true",
        help="Alias for --cpu.",
    )

    return p.parse_args()


def resolve_single_dir(
    spec: TaskSpec,
    override_dir: Optional[str],
) -> Optional[List[str]]:
    """
    Resolve one normal task directory.
    """
    if override_dir:
        if os.path.isdir(override_dir):
            return [override_dir]

        log_info(
            f"Warning: {spec.name} directory not found: {override_dir}"
        )
        return None

    default_dir = os.path.join(
        DEFAULT_WTS_ROOT,
        spec.default_subdirs[0],
    )
    if os.path.isdir(default_dir):
        return [default_dir]

    return None


def resolve_multi_dirs(
    spec: TaskSpec,
    override_dirs: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Resolve a task made of several checkpoint directories.
    """
    if override_dirs:
        found: List[str] = []

        for path in override_dirs:
            if os.path.isdir(path):
                found.append(path)
            else:
                log_info(
                    f"Warning: {spec.name} directory not found: {path}"
                )

        return found or None

    found = []
    for subdir in spec.default_subdirs:
        path = os.path.join(DEFAULT_WTS_ROOT, subdir)
        if os.path.isdir(path):
            found.append(path)

    return found or None


def resolve_task_dirs(
    args: argparse.Namespace,
) -> Dict[str, List[str]]:
    """
    Find all enabled task directories, but do not load any models yet.
    """
    enabled: Dict[str, List[str]] = {}

    for spec in TASK_SPECS:
        override = getattr(args, spec.cli_attr)

        if spec.multi_dir:
            task_dirs = resolve_multi_dirs(spec, override)
        else:
            task_dirs = resolve_single_dir(spec, override)

        if task_dirs:
            enabled[spec.name] = task_dirs

    return enabled


def build_task_checkpoint_counts(
    enabled_tasks: Dict[str, List[str]],
) -> Dict[str, int]:
    """
    Count checkpoints once so the status messages can say something useful.
    """
    counts: Dict[str, int] = {}

    for task_name, task_dirs in enabled_tasks.items():
        counts[task_name] = sum(
            count_checkpoints(ckpt_dir)
            for ckpt_dir in task_dirs
        )

    return counts


def choose_initial_device(force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------
#
# The report is meant to be easy to scan. LigCys appears first because it is
# the main cysteine task of interest.
#

def compute_task_ranks(
    task_pos_probs: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[int, int]]:
    """
    Convert probabilities into 1-based ranks for score-ranked tasks.

    LigCys special case:
      - do NOT assign a rank to residues whose LigCys score is 0.0
    """
    task_ranks: Dict[str, Dict[int, int]] = {}

    for task_name in TOPN_TASKS:
        posprob = task_pos_probs.get(task_name)
        if not posprob:
            continue

        items = sorted(
            posprob.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        if task_name == "LigCys":
            items = [(pos, prob) for pos, prob in items if prob > 0.0]

        task_ranks[task_name] = {
            pos: rank + 1
            for rank, (pos, _) in enumerate(items)
        }

    return task_ranks

def choose_row_positions(
    task_pos_probs: Dict[str, Dict[int, float]],
    ligbind_topk: Optional[int],
) -> List[int]:
    """
    Decide which rows should be printed.

    Cys-only tasks contribute all of their scored positions.
    LigBind only contributes its top-N rows.
    """
    row_positions = set()

    row_positions |= set(task_pos_probs.get("LigCys", {}).keys())
    row_positions |= set(task_pos_probs.get("SSBind", {}).keys())
    row_positions |= set(task_pos_probs.get("ZnBind", {}).keys())
    row_positions |= set(task_pos_probs.get("CuBind", {}).keys())
    row_positions |= set(task_pos_probs.get("FeBind", {}).keys())
    row_positions |= set(task_pos_probs.get("FeSBind", {}).keys())
    row_positions |= set(task_pos_probs.get("HemeBind", {}).keys())

    if "LigBind" in task_pos_probs:
        lb_items = sorted(
            task_pos_probs["LigBind"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if ligbind_topk is None:
            k_val = 20
        else:
            k_val = max(0, int(ligbind_topk))

        row_positions |= {pos for pos, _ in lb_items[:k_val]}

    return sorted(row_positions)


def write_prediction_table(
    seq: str,
    task_pos_probs: Dict[str, Dict[int, float]],
    ligcys_confidence: Dict[int, float],
    dest,
    ligbind_topk: Optional[int],
) -> None:
    """
    Emit one wide residue table for one sequence.

    Column policy:
      - LigCys: percentage score + topN + confidence
      - SSBind / ZnBind: binary only
      - CuBind / FeBind / FeSBind / HemeBind: score + binary
      - LigBind: score + topN
      - any Cys-only task prints "." on non-C rows
    """
    if not task_pos_probs:
        print("No ROI residues found for any task.", file=dest)
        return

    row_positions = choose_row_positions(task_pos_probs, ligbind_topk)
    if not row_positions:
        print("No positions remain after filtering.", file=dest)
        return

    seq_u = seq.strip().upper()
    task_ranks = compute_task_ranks(task_pos_probs)

    header_cols = [
        "pos",
        "AA",
        "LigCys",
        "LigCys_topN",
        "LigCys_confidence",
        "SSBind",
        "SSBind_bin",
        "ZnBind",
        "ZnBind_bin",
        "CuBind",
        "CuBind_bin",
        "FeBind",
        "FeBind_bin",
        "FeSBind",
        "FeSBind_bin",
        "HemeBind",
        "HemeBind_bin",
        "LigBind",
        "LigBind_topN",
    ]

    print("\n", file=dest)
    print("\t".join(header_cols), file=dest)

    for pos in row_positions:
        aa = seq_u[pos - 1] if 1 <= pos <= len(seq_u) else "X"
        row = [str(pos), aa]

        # --------------------------------------------------------------
        # LigCys: percentage score + rank + confidence
        # Confidence is only shown for cysteine rows with score > 0.
        # --------------------------------------------------------------
        lc_probs = task_pos_probs.get("LigCys", {})
        lc_ranks = task_ranks.get("LigCys", {})

        if "LigCys" in CYS_TASKS and aa != "C":
            row.extend([".", ".", "."])
        else:
            lc_prob = lc_probs.get(pos)
            lc_rank = lc_ranks.get(pos)
            lc_conf = ligcys_confidence.get(pos)

            if lc_prob is None:
                row.append("NA")
                row.append(str(lc_rank) if lc_rank is not None else "NA")
                row.append("NA")
            else:
                row.append(f"{100.0 * lc_prob:.2f}%")
                row.append(str(lc_rank) if lc_rank is not None else "NA")

                if lc_prob > 0.0 and lc_conf is not None:
                    row.append(f"{lc_conf:.4f}")
                else:
                    row.append(".")

        # --------------------------------------------------------------
        # Aux models: score with 2 decimals + binary
        # --------------------------------------------------------------
        for task_name in AUX_SCORE_BINARY_TASKS:
            probs = task_pos_probs.get(task_name, {})

            if task_name in CYS_TASKS and aa != "C":
                row.extend([".", "."])
                continue

            prob = probs.get(pos)
            if prob is None:
                row.extend(["NA", "NA"])
                continue

            row.append(f"{prob:.2f}")
            row.append(
                "Yes"
                if prob >= BINARY_THRESHOLDS[task_name]
                else "No"
            )

        # --------------------------------------------------------------
        # LigBind: score + rank
        # --------------------------------------------------------------
        lb_probs = task_pos_probs.get("LigBind", {})
        lb_ranks = task_ranks.get("LigBind", {})

        lb_prob = lb_probs.get(pos)
        lb_rank = lb_ranks.get(pos)

        row.append(f"{lb_prob:.4f}" if lb_prob is not None else "NA")
        row.append(str(lb_rank) if lb_rank is not None else "NA")

        print("\t".join(row), file=dest)

# ---------------------------------------------------------------------------
# Main per-sequence execution
# ---------------------------------------------------------------------------
#
# Runtime order:
#   1. extract the shared all-residue embedding
#   2. run LigBind on the full residue set
#   3. build the smaller Cys-only view
#   4. run all Cys-only tasks
#   5. print the final table
#

def run_one_sequence(
    seq_id: str,
    seq: str,
    enabled_tasks: Dict[str, List[str]],
    task_ckpt_counts: Dict[str, int],
    forge_token: str,
    forge_url: str,
    trunto: int,
    ligbind_topk: Optional[int],
    runtime: RuntimeState,
    pbar: tqdm,
    dest,
) -> None:
    """
    Process one sequence end-to-end.
    """
    seq_u = seq.strip().upper()
    task_pos_probs: Dict[str, Dict[int, float]] = {}
    ligcys_confidence: Dict[int, float] = {}

    pbar.set_postfix_str(f"{seq_id}: ESM-C extract")
    log_info(
        f"[{seq_id}] Stage 1/4: extracting one shared ESM-C "
        "per-residue embedding tensor..."
    )

    all_emb_cpu, all_positions = prepare_all_residue_view(
        seq=seq_u,
        forge_token=forge_token,
        forge_url=forge_url,
        trunto=trunto,
    )

    log_info(
        f"[{seq_id}] ESM-C ready: {all_emb_cpu.size(0)} residue rows, "
        f"{all_emb_cpu.size(1)} features per residue."
    )

    if "LigBind" in enabled_tasks:
        log_info(
            f"[{seq_id}] Stage 2/4: running all-residue ensembles..."
        )

    for spec in TASK_SPECS:
        if spec.name not in enabled_tasks:
            continue
        if spec.name not in ALL_RES_TASKS:
            continue

        pbar.set_postfix_str(f"{seq_id}: {spec.name}")
        log_info(
            f"[{seq_id}] Running {spec.name} "
            f"({task_ckpt_counts[spec.name]} checkpoints, "
            f"all residues, device={runtime.device.type})..."
        )

        result = run_task_with_fallback(
            spec=spec,
            seq_id=seq_id,
            task_dirs=enabled_tasks[spec.name],
            emb_cpu=all_emb_cpu,
            positions=all_positions,
            runtime=runtime,
        )

        task_pos_probs[spec.name] = result.scores

        log_info(
            f"[{seq_id}] Finished {spec.name}: scored "
            f"{len(result.scores)} residue(s)."
        )

    need_cys = any(name in enabled_tasks for name in CYS_TASKS)
    cys_emb_cpu = None
    cys_positions = None

    if need_cys:
        pbar.set_postfix_str(f"{seq_id}: build Cys view")
        log_info(
            f"[{seq_id}] Stage 3/4: building the smaller "
            "cysteine-only tensor..."
        )

        cys_emb_cpu, cys_positions = prepare_cys_view(
            seq=seq_u,
            all_emb_cpu=all_emb_cpu,
            all_positions=all_positions,
        )

        if cys_positions.size == 0:
            log_info(
                f"[{seq_id}] No cysteine residues found. "
                "Cys-only tasks will return no rows."
            )
        else:
            log_info(
                f"[{seq_id}] Cys-only view ready: "
                f"{cys_emb_cpu.size(0)} cysteine residue(s)."
            )

        log_info(
            f"[{seq_id}] Stage 4/4: running cysteine-only ensembles..."
        )

        for spec in TASK_SPECS:
            if spec.name not in enabled_tasks:
                continue
            if spec.name not in CYS_TASKS:
                continue

            pbar.set_postfix_str(f"{seq_id}: {spec.name}")
            log_info(
                f"[{seq_id}] Running {spec.name} "
                f"({task_ckpt_counts[spec.name]} checkpoints, "
                f"cysteines only, device={runtime.device.type})..."
            )

            result = run_task_with_fallback(
                spec=spec,
                seq_id=seq_id,
                task_dirs=enabled_tasks[spec.name],
                emb_cpu=cys_emb_cpu,
                positions=cys_positions,
                runtime=runtime,
            )

            task_pos_probs[spec.name] = result.scores
            if spec.name == "LigCys" and result.confidence is not None:
                ligcys_confidence = result.confidence

            log_info(
                f"[{seq_id}] Finished {spec.name}: scored "
                f"{len(result.scores)} residue(s)."
            )

    pbar.set_postfix_str(f"{seq_id}: write output")
    log_info(f"[{seq_id}] Writing final prediction table...")

    write_prediction_table(
        seq=seq_u,
        task_pos_probs=task_pos_probs,
        ligcys_confidence=ligcys_confidence,
        dest=dest,
        ligbind_topk=ligbind_topk,
    )

    log_info(f"[{seq_id}] Sequence complete.")

    del cys_positions
    del cys_emb_cpu
    del all_positions
    del all_emb_cpu

    if runtime.device.type == "cuda":
        clear_cuda_state()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cmd_str = " ".join(shlex.quote(arg) for arg in sys.argv)
    print(
        "    " + cmd_str + "\n\n"
        "     ------------------------------------------------",
        file=sys.stderr,
    )

    if not args.sequence and not args.fasta:
        raise SystemExit("You must provide either --sequence or --fasta")

    log_info("AiPP startup: resolving model directories...")

    enabled_tasks = resolve_task_dirs(args)
    if not enabled_tasks:
        raise SystemExit(
            "No ensemble directories found. Provide at least one task "
            "directory or ensure default weights exist under "
            f"{DEFAULT_WTS_ROOT}."
        )

    task_ckpt_counts = build_task_checkpoint_counts(enabled_tasks)

    log_info("Enabled task families:")
    for task_name in OUTPUT_TASK_ORDER:
        if task_name not in enabled_tasks:
            continue

        n_dirs = len(enabled_tasks[task_name])
        n_ckpts = task_ckpt_counts[task_name]
        log_info(
            f"  - {task_name}: {n_dirs} dir(s), {n_ckpts} checkpoint(s)"
        )

    runtime = RuntimeState(
        device=choose_initial_device(args.force_cpu),
    )

    log_info(f"Initial device: {runtime.device.type}")

    if args.forge_token:
        forge_token = get_forge_token(args.forge_token)
        try:
            save_token(TOKEN_FILE, forge_token)
            log_info(f"Saved Forge token to {TOKEN_FILE}")
        except OSError as exc:
            log_info(
                f"Warning: could not save Forge token to "
                f"{TOKEN_FILE}: {exc}"
            )
    else:
        forge_token = load_saved_token(TOKEN_FILE)
        if forge_token:
            log_info(f"Using Forge token from {TOKEN_FILE}")
        else:
            log_info(
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
                log_info(f"Saved Forge token to {TOKEN_FILE}")
            except OSError as exc:
                log_info(
                    f"Warning: could not save Forge token to "
                    f"{TOKEN_FILE}: {exc}"
                )

    fasta_records: List[Tuple[str, str]] = []
    if args.fasta:
        log_info(f"Reading FASTA: {args.fasta}")
        fasta_records = read_fasta(args.fasta)
        log_info(f"Loaded {len(fasta_records)} FASTA record(s).")

    multi_fasta_mode = bool(args.fasta and len(fasta_records) > 1)

    out_dir: Optional[str] = None
    if multi_fasta_mode:
        stamp = datetime.datetime.now().strftime("%b%d_%H%M%S").lower()
        out_dir = f"{stamp}_aipp_out"
        try:
            os.makedirs(out_dir, exist_ok=False)
        except OSError as exc:
            raise SystemExit(
                f"Could not create output directory {out_dir}: {exc}"
            )
        log_info(
            f"Multi-FASTA mode: writing one .aipp file per sequence to "
            f"{out_dir}"
        )

    out_fh = None
    if args.out:
        if multi_fasta_mode:
            raise SystemExit(
                "--out is not supported when FASTA has multiple "
                "sequences; outputs are written to per-sequence "
                ".aipp files in a timestamped directory."
            )

        try:
            out_fh = open(args.out, "w", encoding="utf-8")
        except OSError as exc:
            raise SystemExit(
                f"Could not open output file {args.out}: {exc}"
            )

        log_info(f"Single output file: {args.out}")

    jobs: List[Tuple[str, str, str]] = []

    if args.sequence:
        jobs.append((args.id, args.id, args.sequence))

    if args.fasta:
        for hdr, seq in fasta_records:
            uid = hdr.split()[0]
            jobs.append((uid, hdr, seq))

    if not jobs:
        if out_fh is not None:
            out_fh.close()
        return

    log_info(f"Beginning inference on {len(jobs)} sequence(s)...")

    inf_pbar = tqdm(
        total=len(jobs),
        desc="running AiPP inference",
        unit="seq",
        ncols=80,
        leave=True,
        file=sys.stdout,
    )

    try:
        if multi_fasta_mode and out_dir is not None:
            for idx, (uid, hdr, seq) in enumerate(jobs, start=1):
                out_path = os.path.join(out_dir, f"seq{idx}.aipp")
                log_info(
                    f"[{uid}] Starting sequence {idx}/{len(jobs)}. "
                    f"Output file: {out_path}"
                )

                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(">" + hdr + "\n")
                    fh.write(seq.strip().upper() + "\n\n")

                    run_one_sequence(
                        seq_id=uid,
                        seq=seq,
                        enabled_tasks=enabled_tasks,
                        task_ckpt_counts=task_ckpt_counts,
                        forge_token=forge_token,
                        forge_url=args.forge_url,
                        trunto=args.trunto,
                        ligbind_topk=args.ligbindtopk,
                        runtime=runtime,
                        pbar=inf_pbar,
                        dest=fh,
                    )

                inf_pbar.update(1)
        else:
            for idx, (uid, _hdr, seq) in enumerate(jobs, start=1):
                log_info(
                    f"[{uid}] Starting sequence {idx}/{len(jobs)}..."
                )

                dest = out_fh if out_fh is not None else sys.stdout

                run_one_sequence(
                    seq_id=uid,
                    seq=seq,
                    enabled_tasks=enabled_tasks,
                    task_ckpt_counts=task_ckpt_counts,
                    forge_token=forge_token,
                    forge_url=args.forge_url,
                    trunto=args.trunto,
                    ligbind_topk=args.ligbindtopk,
                    runtime=runtime,
                    pbar=inf_pbar,
                    dest=dest,
                )

                inf_pbar.update(1)
    finally:
        inf_pbar.close()

    if out_fh is not None:
        out_fh.close()
        log_info(f"Output written to: {args.out}")

    if runtime.gpu_fallback_used:
        log_info(
            "Run completed after automatic CUDA->CPU fallback due to "
            "GPU VRAM limits."
        )
    else:
        log_info("Run completed.")

    print("", file=sys.stderr)


if __name__ == "__main__":
    splash()
    main()
