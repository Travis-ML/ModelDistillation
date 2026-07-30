# Lab Progression Roadmap

The Tier 1 arc is complete: Lab 00 (the mathematics and numerics), Lab 01 (the objective,
verified against autograd), Lab 02 (real tokenizers, real logits, alignment grounded against
the ecosystem's loss convention). Everything from Lab 03 onward exists to convert that verified
understanding into **operator experience** — the comfort that comes from having launched,
watched, diagnosed, and judged real distillation runs yourself.

Every lab below runs on a single workstation with 128 GB of unified CPU–GPU memory and roughly
273 GB/s of memory bandwidth (arm64, CUDA compute capability sm_121). Two design rules follow
from that hardware class and govern every sizing decision in this document:

1. **Sort every workload by who is decoding.** Prefill runs two orders of magnitude faster than
   decode on bandwidth-limited hardware (~2,000 tok/s prefill vs ~50 tok/s decode for a
   20B-class model). Teacher prefill is cheap; small-student decode is moderate; large-teacher
   decode is the one expensive thing, to be paid once and cached or avoided.
2. **Weights + optimizer + KV cache must fit in 128 GB with headroom.** Full fine-tuning costs
   ~16 bytes/parameter; bf16 inference costs 2; LoRA collapses the optimizer share.

## How the Tier 2 labs are structured

Each notebook from 03 onward has the same three-part shape:

- **Part A — Pre-flight (executed anywhere).** Memory arithmetic, config validation, cache
  integrity checks, loss-function assertions against `kd_core`. Every claim in Part A carries an
  `assert`. If Part A passes, the run in Part B cannot fail for a *silent* reason — only for a
  visible one.
- **Part B — The run.** The actual training or generation, written against cited library APIs
  with parameters from official documentation, sized for the memory and bandwidth budget above.
- **Part C — The verdict.** Expected output ranges, the failure signatures to watch for, and a
  written judgment: did this run do what the lab claims? Comfort with distillation *is* the
  ability to render this verdict quickly, and Part C is where that skill gets practiced.

Artifacts chain deliberately: caches, checkpoints, and eval sets produced by one lab are inputs
to later ones, so the sequence also teaches pipeline hygiene — nothing expensive is computed
twice.

---

## Lab 03 — Your First Real Run: Classical KD and the Capacity Gap

*The one where a training loop finally executes.*

**Question.** When does a soft target beat a hard label, and how much teacher is too much?

**Setup.** SmolLM2-1.7B-Instruct teacher → SmolLM2-360M student (same family, same tokenizer —
Lab 02's pair, promoted). Full fine-tune of the student with the Hinton objective
(`kd_core.hinton_kd_loss`) on a small instruction corpus. Three arms: hard labels only, soft
targets only, mixed (α sweep). Then the capacity-gap probe: distill 1.7B→135M and 360M→135M and
observe where the larger teacher stops helping (Müller et al.; Beyer et al. on patience and
consistency).

**Budget.** Teacher bf16 ~3.4 GB + student full-FT ~6 GB + activations: comfortable. First run
under an hour; the point is a complete, unhurried lap, not scale.

**New operator skills.** The pre-launch ritual (memory arithmetic asserted before any load),
reading a KD loss curve, checkpoint/resume, before/after measurement with Lab 02's diagnostics
(top-1 agreement, forward KL, ECE) rather than loss alone.

**Done when** the soft-target arm beats hard labels on agreement and calibration at equal steps,
the α sweep reproduces the expected U-shape, and you have written a three-sentence Part C
verdict without looking at the answer key.

**Sources to ground against before writing.** Hinton, Vinyals, Dean (2015); Müller, Kornblith,
Hinton; Beyer et al.; current TRL distillation docs for the config vocabulary.

---

## Lab 04 — The Cached-Logit Pipeline: Off-Policy Distillation at Full Speed

*The one where the teacher gets evicted from memory.*

**Question.** What is the cheapest correct pipeline on bandwidth-limited hardware?

**Setup.** Stage 1: teacher prefills the corpus once and writes top-k log-probabilities to disk
(k chosen by Lab 02 §5's measurement workflow, run for real on this corpus). Stage 2: the
teacher is unloaded, and the student trains against the cache with `topk_forward_kl` — all 128
GB now belong to the student. Integrity checks are the heart of Part A: cache↔corpus alignment
spot-verified by re-running the teacher on sampled positions, mass bookkeeping, dtype and
index-width validation.

**Budget.** Caching is pure prefill — pay once, minutes per million tokens. Storage priced by
`bytes_per_token_cache` before a single token is generated. Training then runs teacher-free.

**New operator skills.** Restartable multi-stage pipelines, disk budgeting, cache formats
(memory-mapped tensors), the discipline of validating a purchased asset before training on it.

**Artifacts out.** The logit cache and the trained student checkpoint — both reused by Labs 05
and 07.

**Done when** cached-logit training matches live-teacher training (Lab 03 arm) within noise on
agreement and KL, at materially higher tokens/sec — and you can state the k-vs-bias trade you
made and why.

**Sources.** TRL `DistillationTrainer` docs (`loss_top_k`, and the constraint that a remote
teacher with `beta > 0` forces `loss_top_k = 1`); vLLM logprobs API documentation.

---

## Lab 05 — Divergence Choice: One Variable, Measured Honestly

*The one that teaches ablation discipline.*

**Question.** Forward KL, reverse KL, JSD — what does each actually cost you on a real model?

**Setup.** Lab 04's pipeline rerun with exactly one variable moving: `beta ∈ {0, 0.5, 1}`, then
`T ∈ {1, 2}`, fixed seeds, fixed data, fixed steps. Measured on generations, not just losses:
mean entropy, distinct-n, self-BLEU, agreement — the mode-covering/mode-seeking picture from
Lab 01 §4, now with a 360M student instead of a cartoon bimodal toy.

**Budget.** Six short runs of Lab 03 scale. Cheap individually; the lab's real cost is the
honesty overhead — run matrices, seed handling, a results table that another person could audit.

**New operator skills.** Designing ablations that isolate one cause, multi-seed variance
awareness, judging diversity/quality trade-offs from sampled generations rather than metrics
alone.

**Done when** your results table shows the predicted ordering (reverse KL: lowest entropy,
least diverse, most confident; forward KL: the opposite; JSD between) *or* it doesn't — and
Part C explains which, because "the prediction failed and I know why" is the more valuable
outcome.

**Sources.** Agarwal et al. (GKD, 2306.13649); MiniLLM; the f-divergence organisation from the
2026 on-policy survey.

---

## Lab 06 — Sequence-Level and Black-Box Distillation

*The one where you only have text.*

**Question.** How do you distill when logits are unavailable — and what does teacher-generated
data really cost?

**Setup.** Two regimes. SeqKD (Kim & Rush): the teacher *generates* the training corpus — the
one genuinely expensive pattern on this hardware class, priced explicitly in Part A before
launch (large-model decode at tens of tok/s), then made affordable by keeping the generating
teacher at 1.7B and capping the corpus. Black-box KD: hard SFT on teacher traces with no logits
at all — the DeepSeek-R1-Distill pattern — using a slice of a published teacher-trace dataset
as the "purchased asset" the README's economics call for. Both compared against Lab 04's
token-level student at matched training compute.

**New operator skills.** Decode-cost estimation *before* committing (the budget-first habit),
working with published distillation corpora, knowing when logit access is worth engineering for
and when text is enough.

**Done when** you can rank token-level KD, SeqKD, and trace-SFT on this setup by quality per
unit of teacher compute, with the arithmetic to defend the ranking.

**Sources.** Kim & Rush (2016); DeepSeek-R1 report (distill section); a published trace dataset
card.

---

## Lab 07 — On-Policy Distillation

*The one you babysit.*

**Question.** What does training on the student's own mistakes buy, and what new failure modes
does it open?

**Setup.** TRL's GKD/distillation trainer in on-policy mode: the *student* (cheap, small)
decodes rollouts; the teacher only scores them with prefill — the workload sort from rule 1
working in your favor. Arms: `lmbda ∈ {0, 0.5, 1}` (fraction of on-policy data), cold start
from base vs from Lab 04's distilled checkpoint. Live diagnostics every N steps:
`mean_entropy` on rollouts (Lab 00 §9's estimator-variance story is now a *practical* concern),
generation samples printed mid-run, an explicit abort criterion for entropy collapse.

**Budget.** Student-decode bound: a sub-1B student decodes fast enough for real iteration.
Teacher stays in bf16 for scoring; both fit with room to spare.

**New operator skills.** Babysitting a live loop: reading rollouts mid-training, distinguishing
healthy entropy decline from collapse, early-stopping on diagnostics rather than on loss, the
exposure-bias argument made concrete by watching off-policy and on-policy students fail
differently on the same prompts.

**Done when** you have caught at least one degenerate run *while it was happening* (the lab
manufactures one via an aggressive config) and can articulate the cold-start effect from your
own two arms.

**Sources.** Agarwal et al. (GKD); the 2026 on-policy survey's failure-mode sections
(sampled-token estimator, entropy/length collapse, off-policy cold start).

---

## Lab 08 — Engineering and Scale: The Teacher Server

*The one where you measure instead of quoting.*

**Question.** How do you serve a teacher several times your student's size while training, and
what are this machine's actual numbers?

**Setup.** A vLLM teacher server in its own container serving a quantized 20–32B-class teacher;
the student trains in a separate process, scoring rollouts remotely (TRL's external
teacher-server path, including its documented constraints — no fused-kernel path, `loss_top_k`
forced to 1 when `beta > 0` remotely). Part B's centerpiece is measurement: *your own* prefill
and decode throughput curves vs batch size and sequence length, KV-cache growth against the
memory ceiling, tokens/sec at the student under live remote scoring. The 40:1 prefill:decode
ratio stops being a number this course told you and becomes one you measured.

**New operator skills.** Multi-process orchestration, quantized serving, throughput profiling,
capacity planning with your own curves, reading a serving stack's constraints from its
documentation before designing around it.

**Artifacts out.** A machine profile (your measured curves) that Labs 09–12 size against.

**Sources.** vLLM serving docs; TRL teacher-server docs; quantization method docs for the
chosen format.

---

## Lab 09 — Student Initialisation: Prune-Then-Distill

*The one with checkpoint surgery.*

**Question.** Why start from random when the teacher already contains a student?

**Setup.** Depth-prune the 1.7B teacher to student size by measured layer importance
(Minitron-style: rank layers by the damage their removal does on a probe set, drop the least
important), then distill into the pruned model. Three inits at a fixed distillation budget:
random, pretrained-small (SmolLM2-360M), pruned-teacher. Sheared-LLaMA as the
structured-pruning contrast in the reading.

**New operator skills.** State-dict surgery, layer-importance measurement, evaluating an
initialisation *before* spending the training budget on it, the compute economics of
prune+distill vs train-from-scratch.

**Done when** the pruned-init student wins (or loses) at the fixed budget and your Part C
verdict prices what a pretrained-small init is actually worth in distillation steps.

**Sources.** Minitron (2407.14679); Sheared-LLaMA; DistilBERT/TinyBERT for the layer-selection
lineage.

---

## Lab 10 — Cross-Tokenizer and Representation Distillation

*The one where the vocabularies don't match.*

**Question.** How do you distill across tokenizer families — the situation Lab 02 §6 proved has
no position-wise alignment?

**Setup.** Implement ULD's sorted-logit matching from the paper, assertion-first: Part A
verifies the loss on synthetic cases with known answers (Lab 02 exercise 5 grown up) before any
training. Then a short cross-family run (e.g., a Llama-family teacher → Qwen-family student)
with the GOLD trainer path as the library comparison. Separately: hidden-state matching with a
learned projector on the same-family pair, the DistilBERT/TinyBERT feature-matching idea at
generative scale.

**New operator skills.** Implementing a loss from a paper and proving it correct before trusting
it — the single most SME-defining skill in the sequence — plus projector design and where
representation matching helps beyond logit matching.

**Sources.** ULD; GOLD trainer docs; TinyBERT.

---

## Lab 11 — Evaluation, Failure Modes, and Security

*The one that decides whether any of it worked.*

**Question.** How do you judge a distilled model — and what can distillation silently carry or
leak?

**Setup.** Three movements. First, an eval harness: a standard benchmark subset plus custom
probes, calibration and entropy trajectories, and a contamination check between the distill
corpus and the eval set. Second, a failure-mode gallery built from this course's own artifacts:
replay the entropy-collapse run Lab 07 caught, demonstrate length collapse and diversity
collapse on saved checkpoints, show a student that improved on agreement while degrading on
calibration. Third, the security lens, evaluation-focused: measure whether a planted benign
marker behavior survives distillation (the T-MTB question of *when* backdoors transfer), and
frame distillation-based extraction and its defenses (DistillGuard) from the defender's side.

**New operator skills.** Building an eval you trust, auditing for contamination, adversarial
thinking about your own pipeline, writing a model card-grade assessment.

**Done when** you have a reusable harness, a documented gallery of failures you can recognize
on sight, and a written security assessment of one of your own students.

**Sources.** T-MTB; DistillGuard; lm-evaluation-harness docs.

---

## Lab 12 — Capstone: A Reproducible Distillation Study

*The one you could publish.*

**Question.** One question of your choosing, answered to a standard another researcher could
audit.

**Setup.** Pick a genuine open question the course has equipped you to test — the default
suggestion: *off-policy cached-logit vs on-policy distillation at matched total compute, across
two student sizes.* Pre-register the protocol in the notebook before running: hypotheses, arms,
seeds, metrics, stopping rules. Execute with the Lab 08 machine profile doing the sizing, the
Lab 11 harness doing the judging, and Labs 04/07 pipelines doing the work. Produce the report:
tables, figures, a limitations section, and a manifest (configs, seeds, artifact hashes) that
makes the study rerunnable from scratch.

**Done when** a competent stranger could rerun your study from the notebook alone and get your
conclusion — at which point the course has nothing left to teach you, because you are doing the
thing SMEs do.

---

## The comfort curve, explicitly

The sequence is ordered by what it asks of you as an operator, not by paper chronology:

Labs 03–04 make you someone who has *run* distillation. Labs 05–06 make you someone who can
*compare* methods without fooling yourself. Labs 07–08 make you someone who can run the
*hard* configurations — live loops and multi-process serving — and measure the machine
underneath them. Labs 09–10 make you someone who can *modify* the method: new inits, new
losses, implemented from papers and proven correct. Labs 11–12 make you someone whose
*judgment* can be trusted — which is the actual definition of the expertise this course is for.

Build order matches lab order; each grounding pass (fetching and reading the cited sources)
happens immediately before its lab is written, per the course rule.
