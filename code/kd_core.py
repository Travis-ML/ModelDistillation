"""
kd_core: the distillation primitives, written against real logit tensors.

Design rule for this file: every function takes ordinary [B, T, V] logits and a
[B, T] boolean mask. Nothing here knows or cares whether those logits came from
a Hugging Face model, a vLLM logprob payload, or a toy transformer. That is
deliberate. The objective is the part you must get exactly right, and it is the
part that survives every change of framework.

Everything in here is exercised by lab 01, which asserts each identity against
autograd or against a closed-form value rather than eyeballing a curve.

Convention warning, read once and remember it
---------------------------------------------
`beta` in gjsd() follows TRL's convention (DistillationConfig, GKDConfig):

    beta = 0.0  ->  forward KL,  KL(teacher || student),  mode covering
    beta = 1.0  ->  reverse KL,  KL(student || teacher),  mode seeking

Some write-ups of the GKD paper label these the other way round. Setting beta
backwards trains the opposite objective and produces no error, no warning, and
a perfectly normal-looking loss curve. Lab 01 verifies the direction
numerically; do the same whenever you adopt someone else's config.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "masked_mean", "kl_divergence", "gjsd", "tvd", "hinton_kd_loss",
    "topk_forward_kl", "make_topk_cache", "topk_truncation_bias",
    "shift_for_next_token", "completion_mask_from_prompt_lens",
    "onpolicy_mask", "sequence_logprob",
    "top1_agreement", "mean_entropy", "expected_calibration_error",
    "distinct_n", "self_bleu", "bytes_per_token_cache",
]

EPS = 1e-9


# ---------------------------------------------------------------------------
# Masking and alignment
#
# Nearly every real distillation bug I have seen is here rather than in the
# divergence. The model's distribution at position t predicts the token at
# position t+1, so any mask defined over *tokens* must be shifted before it can
# be used over *predictions*.
# ---------------------------------------------------------------------------

def masked_mean(per_position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a [B, T] quantity over the True entries of a [B, T] mask.

    Note this is a per-token mean, not a per-sequence mean. If your batch mixes
    long and short completions, a per-token mean silently weights the long ones
    more heavily. That is usually what you want for distillation and usually
    not what you want when reporting per-example metrics.

    Masked-out positions are zeroed by selection, not by multiplication. A
    padded position can legitimately hold inf or NaN (nothing constrains the
    model's output where nothing is supervised), and 0 * inf is NaN, so
    multiplying by the mask would let a position you excluded poison the mean.
    """
    m = mask.to(per_position.dtype)
    per_position = torch.where(mask.bool(), per_position,
                               torch.zeros((), dtype=per_position.dtype,
                                           device=per_position.device))
    return (per_position * m).sum() / m.sum().clamp_min(1.0)


def shift_for_next_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align logits and mask for next-token prediction.

    Given logits over positions 0..T-1 and a mask marking which *tokens* are
    supervised, returns (student, teacher, mask) trimmed so that entry t of the
    logits is scored against entry t of the mask, both referring to the
    prediction of token t+1.
    """
    return (student_logits[:, :-1], teacher_logits[:, :-1], token_mask[:, 1:])


def completion_mask_from_prompt_lens(
    input_ids: torch.Tensor, prompt_lens: Sequence[int],
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Mask marking completion tokens only, excluding prompt and padding.

    Distilling on prompt tokens dilutes the gradient with a region the student
    is never asked to produce, and it inflates any agreement number you report,
    because both models trivially agree on copying a prompt.
    """
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device)[None, :]
    lens = torch.tensor(prompt_lens, device=input_ids.device)[:, None]
    mask = pos >= lens
    if pad_token_id is not None:
        mask = mask & (input_ids != pad_token_id)
    return mask


def onpolicy_mask(generated: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Mask for on-policy rollouts: True on generated tokens up to and
    including the first EOS, False afterwards.

    Training past EOS teaches the student to model padding, which shows up
    later as a model that will not stop.
    """
    is_eos = generated == eos_token_id
    after_eos = is_eos.cumsum(dim=1) - is_eos.long() > 0
    return ~after_eos


def sequence_logprob(logits: torch.Tensor, tokens: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Summed log p(token) per sequence. Used for rejection sampling, for
    reward-weighted variants, and for length-normalised diagnostics."""
    lp = F.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask.to(lp.dtype)).sum(dim=1)


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------

def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    T: float = 1.0,
    direction: str = "forward",
    scale_by_T2: bool = True,
) -> torch.Tensor:
    """Token-level KL between temperature-softened distributions.

    direction="forward"  KL(teacher || student). Mode covering, zero avoiding.
                         The student is punished for putting no mass where the
                         teacher has mass, so a low-capacity student smears.
    direction="reverse"  KL(student || teacher). Mode seeking, zero forcing.
                         The student concentrates on teacher modes and abandons
                         the tail, which is usually what you want for
                         generation and usually costs you diversity.

    The T**2 factor compensates for the 1/T that appears in the gradient of the
    softened loss. Keep it whenever you mix this term with a hard-label
    cross-entropy at T=1, or the relative weighting of the two terms will drift
    every time you change T.
    """
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    if direction == "forward":
        p_log, q_log = t_log, s_log
    elif direction == "reverse":
        p_log, q_log = s_log, t_log
    else:
        raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")
    per_pos = (p_log.exp() * (p_log - q_log)).sum(-1)
    out = masked_mean(per_pos, mask)
    return out * (T ** 2) if scale_by_T2 else out


def gjsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    T: float = 1.0,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence, TRL's beta convention.

        JSD_beta(p || q) = beta * KL(p || m) + (1 - beta) * KL(q || m)
        m = beta * p + (1 - beta) * q      p = teacher, q = student

    beta -> 0 : proportional to forward KL(teacher || student)
    beta -> 1 : proportional to reverse KL(student || teacher)
    beta = 0.5: symmetric JSD, bounded, the usual safe default when the student
                is far too small to cover the teacher.

    Bounded divergences matter more than they sound. Under KL a single token
    where the student assigns near-zero probability to something the teacher
    likes can dominate an entire batch's gradient. JSD and TVD cannot.
    """
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    q, p = s_log.exp(), t_log.exp()
    m_log = (beta * p + (1 - beta) * q).clamp_min(EPS).log()
    per_pos = beta * (p * (t_log - m_log)).sum(-1) + (1 - beta) * (q * (s_log - m_log)).sum(-1)
    return masked_mean(per_pos, mask)


def tvd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        mask: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Total variation distance, bounded in [0, 1] per position."""
    q = F.softmax(student_logits / T, dim=-1)
    p = F.softmax(teacher_logits / T, dim=-1)
    return masked_mean(0.5 * (p - q).abs().sum(-1), mask)


def hinton_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    T: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """The 2015 objective: alpha * T^2 * KL(soft) + (1 - alpha) * CE(hard).

    alpha = 1.0 is pure distillation. Keep a nonzero hard-label term whenever
    you have trustworthy labels and a teacher that is merely good rather than
    correct, since it bounds how far the student can follow a confident
    teacher error.
    """
    soft = kl_divergence(student_logits, teacher_logits, mask, T=T,
                         direction="forward", scale_by_T2=True)
    if alpha >= 1.0:
        return soft
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1), ignore_index=-100,
    )
    return alpha * soft + (1 - alpha) * ce


# ---------------------------------------------------------------------------
# Top-k logit caching
#
# A 151k-vocabulary teacher in bf16 costs ~302 KB per token position if you
# store the dense distribution. One million cached tokens is ~302 GB. Nobody
# stores dense. Everybody stores top-k, and almost nobody measures the bias
# that introduces. Lab 06 measures it.
# ---------------------------------------------------------------------------

def make_topk_cache(teacher_logits: torch.Tensor, k: int = 32,
                    T: float = 1.0) -> dict[str, torch.Tensor]:
    """Compress a dense teacher distribution to top-k plus an explicit tail.

    Returns log-probabilities, not logits, because that is what an inference
    server will hand you and because it removes any ambiguity about which
    temperature the normalisation was performed at.
    """
    log_p = F.log_softmax(teacher_logits / T, dim=-1)
    top_lp, top_idx = log_p.topk(k, dim=-1)
    tail = (1.0 - top_lp.exp().sum(-1)).clamp_min(0.0)
    return {"topk_logprobs": top_lp, "topk_idx": top_idx,
            "tail_logprob": tail.clamp_min(EPS).log(), "k": torch.tensor(k)}


def topk_forward_kl(
    student_logits: torch.Tensor,
    cache: dict[str, torch.Tensor],
    mask: torch.Tensor,
    T: float = 1.0,
    use_tail: bool = True,
    scale_by_T2: bool = True,
) -> torch.Tensor:
    """Forward KL when only the teacher's top-k log-probabilities were kept.

    Two ways to handle the missing mass:

    use_tail=False  renormalise over the k retained entries. Simple, and it
                    systematically understates the divergence because it
                    pretends the teacher never considered anything else.
    use_tail=True   treat everything outside the top-k as one aggregate bucket
                    and match the student's total mass on that bucket. Costs
                    one extra term and removes most of the bias.

    Everything here is computed in fp32 regardless of the student's dtype. This
    is not defensive cargo-culting; the tail term genuinely breaks in bf16. The
    student's mass on the teacher's top-k routinely exceeds 0.996 on
    teacher-forced text, bfloat16 has 8 mantissa bits so its resolution near 1.0
    is about 0.0039, and both that sum and the `1 - 1e-6` clamp bound below
    round to exactly 1.0. log1p(-1.0) is -inf, the tail term becomes +inf, and
    the loss is NaN from the first step onward. Casting up front costs one copy
    of a [B, T, k] tensor and matches what TopKCacheWriter already does.
    """
    s_log = F.log_softmax(student_logits.float() / T, dim=-1)
    top_lp, top_idx = cache["topk_logprobs"].float(), cache["topk_idx"]
    s_top = s_log.gather(-1, top_idx)
    if not use_tail:
        p_log = top_lp - torch.logsumexp(top_lp, dim=-1, keepdim=True)
        per_pos = (p_log.exp() * (p_log - s_top)).sum(-1)
    else:
        per_pos = (top_lp.exp() * (top_lp - s_top)).sum(-1)
        p_tail_log = cache["tail_logprob"].float()
        s_tail_log = torch.log1p(-s_top.exp().sum(-1).clamp(max=1 - 1e-6))
        per_pos = per_pos + p_tail_log.exp() * (p_tail_log - s_tail_log)
    out = masked_mean(per_pos, mask)
    return out * (T ** 2) if scale_by_T2 else out


def topk_truncation_bias(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         mask: torch.Tensor, ks: Sequence[int],
                         T: float = 1.0) -> list[dict]:
    """Exact dense forward KL versus its top-k approximations, per k.

    Run this on a sample of your real corpus before committing to a k. The
    answer depends on how peaked your teacher is, which depends on the domain:
    a math teacher at low temperature is far more compressible than a
    creative-writing teacher at temperature 1.
    """
    dense = float(kl_divergence(student_logits, teacher_logits, mask, T=T,
                                scale_by_T2=False))
    rows = []
    for k in ks:
        cache = make_topk_cache(teacher_logits, k=k, T=T)
        renorm = float(topk_forward_kl(student_logits, cache, mask, T=T,
                                       use_tail=False, scale_by_T2=False))
        tailed = float(topk_forward_kl(student_logits, cache, mask, T=T,
                                       use_tail=True, scale_by_T2=False))
        covered = float(cache["topk_logprobs"].exp().sum(-1)[mask].mean())
        rows.append({"k": k, "dense_kl": dense, "renorm_kl": renorm,
                     "tail_bucket_kl": tailed, "mean_mass_covered": covered,
                     "renorm_rel_err": (renorm - dense) / max(dense, EPS),
                     "tail_rel_err": (tailed - dense) / max(dense, EPS)})
    return rows


def bytes_per_token_cache(vocab_size: int, k: int, dtype_bytes: int = 2,
                          idx_bytes: int = 4) -> dict[str, float]:
    """Storage arithmetic for a logit cache. Do this before you generate
    anything, not after you have filled a disk."""
    dense = vocab_size * dtype_bytes
    sparse = k * (dtype_bytes + idx_bytes) + dtype_bytes  # +1 tail value
    return {"dense_bytes_per_token": dense, "topk_bytes_per_token": sparse,
            "compression": dense / sparse,
            "dense_gb_per_1M_tokens": dense * 1e6 / 1e9,
            "topk_gb_per_1M_tokens": sparse * 1e6 / 1e9}


# ---------------------------------------------------------------------------
# Diagnostics
#
# Perplexity against held-out gold text is the wrong headline metric for a
# distillation run. It does not tell you whether the student picks the
# teacher's token, and it is blind to the failure modes that actually bite:
# entropy collapse, diversity collapse, and overconfidence.
# ---------------------------------------------------------------------------

def top1_agreement(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                   mask: torch.Tensor) -> float:
    """Fraction of supervised positions where student and teacher argmax
    agree. The honest headline number for a distillation run."""
    agree = (student_logits.argmax(-1) == teacher_logits.argmax(-1)) & mask
    return float(agree.sum()) / max(1, int(mask.sum()))


def mean_entropy(logits: torch.Tensor, mask: torch.Tensor, T: float = 1.0) -> float:
    """Mean predictive entropy in nats. Log this every on-policy step. A
    monotone decline is entropy collapse, and it precedes the length and
    diversity collapse you will otherwise notice far too late."""
    lp = F.log_softmax(logits / T, dim=-1)
    H = -(lp.exp() * lp).sum(-1)
    return float(masked_mean(H, mask))


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor,
                               mask: torch.Tensor, n_bins: int = 10) -> float:
    """Standard ECE over next-token predictions. Distilled students routinely
    become more confident and less correct at the same time."""
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    conf, correct = conf[mask], correct[mask]
    if conf.numel() == 0:
        return 0.0
    ece, n = 0.0, conf.numel()
    edges = torch.linspace(0, 1, n_bins + 1, device=conf.device)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not bool(sel.any()):
            continue
        ece += float(sel.sum()) / n * abs(float(conf[sel].mean()) - float(correct[sel].mean()))
    return ece


def distinct_n(samples: Sequence[str], n: int = 3) -> float:
    """Unique n-grams over total n-grams across a set of samples."""
    grams, total = set(), 0
    for s in samples:
        toks = s.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return len(grams) / max(1, total)


def self_bleu(samples: Sequence[str], n: int = 3) -> float:
    """Mean n-gram precision of each sample against all the others. Higher
    means the samples resemble each other more, i.e. less diverse. Pair it
    with distinct_n; they fail in different ways."""
    if len(samples) < 2:
        return 0.0
    grams = [set(tuple(s.split()[i:i + n]) for i in range(max(0, len(s.split()) - n + 1)))
             for s in samples]
    scores = []
    for i, s in enumerate(samples):
        toks = s.split()
        g = [tuple(toks[j:j + n]) for j in range(len(toks) - n + 1)]
        if not g:
            continue
        ref = set().union(*[grams[j] for j in range(len(samples)) if j != i])
        scores.append(sum(1 for x in g if x in ref) / len(g))
    return sum(scores) / max(1, len(scores))
