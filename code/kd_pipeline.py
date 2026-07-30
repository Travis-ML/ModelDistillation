"""
kd_pipeline: the engineering layer for the Tier 2 labs.

kd_core answers "is the objective right?"; this file answers "will the run
survive?" Everything here is either pure arithmetic or pure bookkeeping, which
means everything here is testable without a GPU — and the Tier 2 labs' Part A
sections do exactly that, asserting each piece before any training begins.

Design rule: nothing in this file imports a model. Budgets take parameter
counts, caches take tensors, monitors take floats. The expensive objects stay
in the notebooks where you can see them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "set_seed_everywhere", "config_fingerprint",
    "full_ft_gb", "lora_ft_gb", "infer_gb", "kv_cache_gb",
    "bandwidth_bound_decode_tps", "decode_wallclock_hours", "prefill_wallclock_hours",
    "MemoryPlan",
    "TopKCacheWriter", "TopKCacheReader",
    "EntropyMonitor",
    "uld_sorted_loss",
    "RunManifest",
]


# ---------------------------------------------------------------------------
# Reproducibility and identity
# ---------------------------------------------------------------------------

def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_fingerprint(cfg: dict) -> str:
    """Short stable hash of a config dict. Two runs with the same fingerprint
    and the same seed should be the same experiment; put it in every output
    filename and you will never again wonder which run produced which file."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Memory and bandwidth arithmetic
#
# The rules the whole course sizes against:
#   full fine-tune  ~16 bytes/param  (bf16 weights+grads, fp32 Adam m+v+master)
#   bf16 inference   ~2 bytes/param
#   decode is bandwidth-bound: every generated token re-reads the weights.
# ---------------------------------------------------------------------------

def full_ft_gb(params_b: float) -> float:
    """Full fine-tuning footprint in GB: weights(2) + grads(2) + Adam fp32
    moments(8) + fp32 master weights(4) = 16 bytes/param."""
    return params_b * 16.0

def lora_ft_gb(params_b: float, trainable_frac: float = 0.01) -> float:
    """LoRA: frozen bf16 base + full 16-byte cost only on the adapter slice."""
    return params_b * 2.0 + params_b * trainable_frac * 16.0

def infer_gb(params_b: float, bytes_per_param: float = 2.0) -> float:
    return params_b * bytes_per_param

def kv_cache_gb(n_layers: int, n_kv_heads: int, head_dim: int,
                seq_len: int, batch: int, bytes_per: int = 2) -> float:
    """K and V, per layer, per position: 2 * n_kv_heads * head_dim values."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per / 1e9

def bandwidth_bound_decode_tps(params_b: float, bw_gbs: float,
                               bytes_per_param: float = 2.0) -> float:
    """Upper bound on decode tokens/sec from weight reads alone: each token
    must stream every parameter through the compute units once."""
    return bw_gbs / (params_b * bytes_per_param)

def decode_wallclock_hours(n_tokens: float, tps: float) -> float:
    return n_tokens / tps / 3600.0

def prefill_wallclock_hours(n_tokens: float, prefill_tps: float) -> float:
    return n_tokens / prefill_tps / 3600.0


@dataclass
class MemoryPlan:
    """Declare the plan, then assert it fits *before* loading anything.

    parts: name -> GB. headroom_frac reserves space for KV cache, activations,
    fragmentation, and the allocator's mood; 0.15 is a floor, not a target.
    """
    total_gb: float
    parts: dict = field(default_factory=dict)
    headroom_frac: float = 0.15

    def add(self, name: str, gb: float) -> "MemoryPlan":
        self.parts[name] = round(gb, 2)
        return self

    @property
    def planned_gb(self) -> float:
        return sum(self.parts.values())

    @property
    def fits(self) -> bool:
        return self.planned_gb <= self.total_gb * (1 - self.headroom_frac)

    def table(self) -> str:
        lines = [f"{'component':<38} {'GB':>8}"]
        for k, v in self.parts.items():
            lines.append(f"{k:<38} {v:>8.2f}")
        lines.append(f"{'-' * 47}")
        lines.append(f"{'planned':<38} {self.planned_gb:>8.2f}")
        lines.append(f"{'budget (after headroom)':<38} "
                     f"{self.total_gb * (1 - self.headroom_frac):>8.2f}")
        lines.append(f"{'fits':<38} {str(self.fits):>8}")
        return "\n".join(lines)

    def assert_fits(self) -> None:
        assert self.fits, (
            f"memory plan does not fit: {self.planned_gb:.1f} GB planned, "
            f"{self.total_gb * (1 - self.headroom_frac):.1f} GB available after "
            f"headroom.\n{self.table()}")


# ---------------------------------------------------------------------------
# The top-k logit cache
#
# Layout on disk (one directory per cache):
#   manifest.json      k, vocab, dtypes, n_rows, T, corpus fingerprint
#   topk_logprobs.npy  [n_rows, T, k]  float16
#   topk_idx.npy       [n_rows, T, k]  int32
#   tail_logprob.npy   [n_rows, T]     float16
#   mask.npy           [n_rows, T]     bool     (supervised positions)
#   input_ids.npy      [n_rows, T]     int32    (kept so alignment is checkable)
#
# The corpus fingerprint is a hash of input_ids. A cache that cannot prove it
# belongs to your corpus is not an asset, it is a liability with good storage
# characteristics.
# ---------------------------------------------------------------------------

class TopKCacheWriter:
    def __init__(self, path: str, k: int, vocab_size: int, seq_len: int,
                 temperature: float = 1.0):
        self.path, self.k, self.vocab_size = path, k, vocab_size
        self.seq_len, self.temperature = seq_len, temperature
        os.makedirs(path, exist_ok=True)
        self._lp, self._idx, self._tail, self._mask, self._ids = [], [], [], [], []

    def append(self, teacher_logits: torch.Tensor, input_ids: torch.Tensor,
               mask: torch.Tensor) -> None:
        """teacher_logits [B,T,V] raw logits; caches log-probs at self.temperature."""
        assert teacher_logits.shape[-1] == self.vocab_size, "vocab mismatch"
        assert teacher_logits.shape[1] == self.seq_len, "seq_len mismatch"
        log_p = F.log_softmax(teacher_logits.float() / self.temperature, dim=-1)
        top_lp, top_idx = log_p.topk(self.k, dim=-1)
        tail = (1.0 - top_lp.exp().sum(-1)).clamp_min(1e-9).log()
        self._lp.append(top_lp.to(torch.float16).cpu())
        self._idx.append(top_idx.to(torch.int32).cpu())
        self._tail.append(tail.to(torch.float16).cpu())
        self._mask.append(mask.bool().cpu())
        self._ids.append(input_ids.to(torch.int32).cpu())

    def finalize(self) -> dict:
        arrs = {
            "topk_logprobs": torch.cat(self._lp).numpy(),
            "topk_idx": torch.cat(self._idx).numpy(),
            "tail_logprob": torch.cat(self._tail).numpy(),
            "mask": torch.cat(self._mask).numpy(),
            "input_ids": torch.cat(self._ids).numpy(),
        }
        for name, a in arrs.items():
            np.save(os.path.join(self.path, f"{name}.npy"), a)
        manifest = {
            "k": self.k, "vocab_size": self.vocab_size, "seq_len": self.seq_len,
            "temperature": self.temperature, "n_rows": int(arrs["mask"].shape[0]),
            "corpus_fingerprint": hashlib.sha256(
                arrs["input_ids"].tobytes()).hexdigest()[:16],
            "bytes_on_disk": int(sum(a.nbytes for a in arrs.values())),
        }
        with open(os.path.join(self.path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


class TopKCacheReader:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "manifest.json")) as f:
            self.manifest = json.load(f)
        mm = lambda n: np.load(os.path.join(path, f"{n}.npy"), mmap_mode="r")
        self._lp, self._idx = mm("topk_logprobs"), mm("topk_idx")
        self._tail, self._mask = mm("tail_logprob"), mm("mask")
        self._ids = mm("input_ids")

    def __len__(self) -> int:
        return self.manifest["n_rows"]

    def verify_against(self, input_ids: np.ndarray) -> None:
        """Refuse to train on a cache that does not match the corpus in hand."""
        fp = hashlib.sha256(np.ascontiguousarray(
            input_ids.astype(np.int32)).tobytes()).hexdigest()[:16]
        assert fp == self.manifest["corpus_fingerprint"], (
            f"cache fingerprint {self.manifest['corpus_fingerprint']} != corpus {fp}: "
            "this cache was built from different token ids. Do not train.")

    def batch(self, rows) -> dict:
        """Returns tensors shaped for kd_core.topk_forward_kl, plus ids/mask."""
        rows = np.asarray(rows)
        return {
            "topk_logprobs": torch.from_numpy(self._lp[rows].copy()).float(),
            "topk_idx": torch.from_numpy(self._idx[rows].copy()).long(),
            "tail_logprob": torch.from_numpy(self._tail[rows].copy()).float(),
            "mask": torch.from_numpy(self._mask[rows].copy()),
            "input_ids": torch.from_numpy(self._ids[rows].copy()).long(),
            "k": torch.tensor(self.manifest["k"]),
        }


# ---------------------------------------------------------------------------
# Live-run diagnostics
# ---------------------------------------------------------------------------

class EntropyMonitor:
    """Tracks mean predictive entropy over training and answers one question:
    is this a healthy decline or a collapse?

    Heuristic, deliberately simple: collapse is flagged when entropy has fallen
    below `floor_nats` OR has lost more than `drop_frac` of its starting value
    within the trailing `window` observations. Tune both on your own runs; the
    point is to have *an* automatic tripwire, because the human watching the
    loss curve reliably notices collapse three hundred steps too late.
    """

    def __init__(self, floor_nats: float = 0.15, drop_frac: float = 0.6,
                 window: int = 20):
        self.floor_nats, self.drop_frac, self.window = floor_nats, drop_frac, window
        self.history: list[tuple[int, float]] = []

    def update(self, step: int, entropy_nats: float) -> None:
        self.history.append((step, float(entropy_nats)))

    @property
    def collapsed(self) -> bool:
        if len(self.history) < 3:
            return False
        latest = self.history[-1][1]
        if latest < self.floor_nats:
            return True
        recent = [h for _, h in self.history[-self.window:]]
        return latest < recent[0] * (1 - self.drop_frac)

    def report(self) -> str:
        if not self.history:
            return "no observations"
        s0, h0 = self.history[0]
        s1, h1 = self.history[-1]
        return (f"entropy {h0:.3f} nats @ step {s0} -> {h1:.3f} @ step {s1} "
                f"({'COLLAPSED' if self.collapsed else 'healthy'})")


# ---------------------------------------------------------------------------
# Cross-tokenizer: the ULD sorted-logit loss
#
# Universal Logit Distillation's move: if two vocabularies cannot be aligned
# index-to-index, compare the probability vectors as *sorted* vectors — the
# multiset of probabilities, stripped of token identity. The L1 distance
# between sorted distributions is well-defined across any two vocabularies
# (pad the shorter sorted vector with zeros) and bounded by 2.
# ---------------------------------------------------------------------------

def uld_sorted_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    student_mask: torch.Tensor, teacher_mask: torch.Tensor,
                    T: float = 1.0) -> torch.Tensor:
    """Mean L1 between sorted next-token distributions over paired positions.

    student_logits [B, Ts, Vs], teacher_logits [B, Tt, Vt]; masks mark each
    side's supervised positions. Positions are paired in order within each
    sequence and truncated to the shorter side's count — the crude alignment
    of the original ULD paper. (GOLD's incremental-decode span matching is the
    principled upgrade; Lab 10 uses the library for that.)
    """
    B = student_logits.shape[0]
    losses = []
    for b in range(B):
        sp = F.softmax(student_logits[b][student_mask[b]] / T, dim=-1)
        tp = F.softmax(teacher_logits[b][teacher_mask[b]] / T, dim=-1)
        n = min(sp.shape[0], tp.shape[0])
        if n == 0:
            continue
        sp, tp = sp[:n], tp[:n]
        ss, _ = sp.sort(dim=-1, descending=True)
        ts, _ = tp.sort(dim=-1, descending=True)
        V = max(ss.shape[-1], ts.shape[-1])
        ss = F.pad(ss, (0, V - ss.shape[-1]))
        ts = F.pad(ts, (0, V - ts.shape[-1]))
        losses.append((ss - ts).abs().sum(-1).mean())
    assert losses, "no supervised positions on either side"
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Everything needed to rerun or audit a run, written next to its outputs."""
    name: str
    config: dict
    seed: int
    artifacts_in: dict = field(default_factory=dict)    # name -> fingerprint
    artifacts_out: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def fingerprint(self) -> str:
        return config_fingerprint({**self.config, "seed": self.seed})

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        out = os.path.join(directory, f"manifest_{self.name}_{self.fingerprint}.json")
        with open(out, "w") as f:
            json.dump({**asdict(self), "fingerprint": self.fingerprint}, f, indent=2)
        return out
