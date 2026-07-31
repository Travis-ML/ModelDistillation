# Distillation of Generative Models

My working lab course on distilling generative models, from the objective function up to a
reproducible research pipeline. The real-model track is sized for the single box I run it on:
128 GB of unified CPU-GPU memory at ~273 GB/s (arm64, CUDA compute capability sm_121).

---

## How these labs are built, and why

Labs come in two tiers, and each one says which it is on the first line.

**Tier 1: executed and asserted.** Runs on any machine in seconds. Works on synthetic logit
tensors, real tokenizers, and cached artifacts. Every claim gets checked with an `assert`
against autograd or a closed form. These labs were run end to end when they were written, so
the outputs in the notebooks are real.

**Tier 2: grounded but not executed.** Real distillation runs on real checkpoints, written
against cited library APIs with parameters from official docs and published recipes, sized for
the memory and bandwidth budget above. These were NOT executed when written. I can't validate a
real training run anywhere but my own hardware, and a green checkmark from some toy stand-in
would just be false confidence. Instead, each Tier 2 lab carries expected output ranges, the
failure signatures to watch for, and a verification checklist (Part A) that runs first.

The rule behind the split: the stuff worth executing is the stuff that needs no training, and
the stuff that needs training can only be validated on my box. The objective, the masking, the
caching arithmetic, the tokenizer alignment. All of it is exact, fast to check, and it's where
the silent bugs actually live. A wrong `beta` or an unshifted mask trains the wrong thing
without throwing an error or even bending the loss curve.

Not hypothetical, either. At one point `kd_core.gjsd` had a docstring describing `beta`
backwards relative to TRL's convention. The code was right and the comment was wrong, which is
the worse combination, because you trust the comment. Caught it by reading the current TRL docs
and settled it by evaluating four numbers. Lab 01 now has that check as a permanent assertion.

---

## Reading the hardware before designing the pipeline

Two measured facts drive every Tier 2 decision.

**Capacity is generous, bandwidth is not.** 128 GB of coherent unified memory means a 70B
teacher fits. 273 GB/s shared across CPU and GPU means it won't be fast. LMSYS measured GPT-OSS
20B in MXFP4 at roughly 2,053 tok/s prefill vs 49.7 tok/s decode on this class of hardware.
That ~40:1 ratio is the number to design around.

**So: sort every workload by who is decoding.**

| Workload | Shape | Who decodes | Cost on this box |
|---|---|---|---|
| Caching teacher top-k logits over a corpus | prefill | nobody | cheap, pay once |
| Teacher scoring student rollouts | prefill | nobody | cheap |
| Student rollouts for on-policy distillation | decode | **the student** (small) | moderate |
| Teacher generating a SeqKD corpus | decode | **the teacher** (large) | expensive |

Flagging a correction here because I got this wrong at first and the mistake is instructive.
Seeing 40:1, my first take was "on-policy distillation is the expensive path on a
bandwidth-pinned box, so do a long off-policy phase and a short on-policy phase." Wrong, and
the error was forgetting which model generates. In on-policy distillation the rollouts come
from the student, the small model. The teacher only ever does a scoring forward pass. A 1.7B
bf16 student is ~3.4 GB, so weight reads alone cap it near 273/3.4 ≈ 80 decode steps/sec, and
batching spreads that across the whole batch.

The actually-expensive pattern is the opposite one: sequence-level KD, where the big teacher
generates the corpus. A 32B bf16 teacher is 64 GB, which caps decode near 4 steps/sec before
anything else even binds.

So on this hardware the usual advice about what's affordable gets inverted:

- Prefer on-policy distillation and cached-logit off-policy training. Both lean on prefill and
  small-model decode.
- Treat teacher-generated corpora as a purchased asset. Generate once and cache it, use an
  existing published teacher-generation dataset, or rent an hour of H100 for that phase.
- Never pay for large-teacher decode twice.

### Memory budget

Full fine-tuning costs about 16 bytes per parameter (bf16 weights and grads, fp32 Adam moments
and master weights). LoRA collapses the optimizer share to almost nothing.

| Configuration | Teacher (bf16 inference) | Student | Total | Verdict |
|---|---|---|---|---|
| 8B teacher + 1.7B student, full FT | 16 GB | 27 GB | ~43 GB | comfortable, good default |
| 14B teacher + 1.7B student, full FT | 28 GB | 27 GB | ~55 GB | comfortable |
| 8B teacher + 4B student, full FT | 16 GB | 64 GB | ~80 GB | workable, watch KV cache |
| 32B teacher + 8B student, LoRA | 64 GB | ~18 GB | ~82 GB | viable, quantize teacher if tight |
| 32B teacher + 4B student, full FT | 64 GB | 64 GB | ~128 GB | won't fit, quantize or offload |

Leave real headroom for KV cache and activations. The table is weights and optimizer state
only.

### arm64 notes

Wheel availability is the recurring friction on sm_121/aarch64. Reuse a known-good vLLM
container for the teacher server instead of building fresh, and verify `flash-attn`,
`bitsandbytes`, and Liger availability before designing a run around them. TRL's teacher-server
path explicitly does not support Liger kernels.

---

## Course spine

The prerequisite refreshers assume you can read PyTorch and have trained a transformer before.
Nothing more.

| # | Unit | Tier | Core question |
|---|---|---|---|
| 00 | Distributions, divergences, and numerics | 1 | What is actually being minimised? |
| 01 | **The objective, verified** | 1 | Why `T²`, why `beta` bites, why `k` is measured |
| 02 | Tokenizers, alignment, and real model outputs | 1 | Where do teacher logits come from and how do they line up? |
| 03 | Classical KD and the capacity gap | 2 | When does a soft target beat a hard label? |
| 04 | Off-policy distillation with cached logits | 2 | The cheapest real pipeline on this box |
| 05 | Divergence choice and mode behaviour | 2 | Forward, reverse, JSD: what each costs you |
| 06 | Sequence-level, rationale, and black-box KD | 2 | Distilling when you only have an API |
| 07 | On-policy distillation | 2 | Exposure bias, GKD, `lmbda`, cold start |
| 08 | Engineering and scale | 2 | Top-k caching, teacher servers, throughput math |
| 09 | Student initialisation and prune-then-distill | 2 | Stop starting from random |
| 10 | Cross-tokenizer and representation distillation | 2 | ULD, GOLD, hidden-state matching |
| 11 | Evaluation, failure modes, and security | 1+2 | Entropy collapse, calibration, backdoor transfer, extraction |
| 12 | Capstone: a reproducible distillation study | 2 | Multi-arm comparison you could publish |

### Status

- [x] `code/kd_core.py`: divergences, top-k caching, alignment, diagnostics. Verified.
- [x] **Lab 00**: executed, 10/10 code cells, all assertions passing. Softmax/logsumexp
      numerics, the H/CE/KL identity, KL not being a metric (while sqrt-JSD is), the
      f-divergence family checked generator-vs-direct, Pinsker and boundedness, the
      bf16/fp16/fp32 loss-computation policy, the whiteboard gradient, and the k1/k2/k3
      sampled KL estimators. One measured surprise worth remembering: k3's variance advantage
      only holds when student and teacher are close, which is Unit 07's cold-start problem in
      miniature.
- [x] **Lab 01**: executed, 12/12 cells, all assertions passing.
- [x] **Lab 02**: executed, 7/7 code cells, all assertions passing. Real tokenizers (GPT-2,
      Qwen2.5, SmolLM2) and a real teacher/student pair (SmolLM2 360M -> 135M). The
      centerpiece: `kd_core`'s external shift+mask reproduces HF's internal
      `model(labels=...).loss` to float precision on a ragged chat-templated batch. Also
      measured on real logits: the two top-k truncation estimators are biased in opposite
      directions (renormalise overstates, tail-bucket understates), so pick `k` by
      measurement, not doctrine. Plus cross-tokenizer boundary overlap and cache/prefill
      pricing.
- [x] **Labs 03 through 12: built.** Every Tier 2 lab follows the three-part shape from
      `ROADMAP.md`. Part A is pre-flight, executed and asserted when written: config
      discipline, memory plans, loss/format/detector verification. Part B is the training run,
      gated behind a `RUN_TRAINING`-style flag I flip on the box, grounded against the
      installed library APIs, not executed when written. Part C is expected ranges, failure
      signatures, the verdict to write, and exercises. Some Part A checks that ran for real:
      Lab 04's cache format survives a tamper attack, Lab 08's remote-scoring client verifies
      end to end against a live mock server, Lab 09 does real depth-pruning surgery on a 135M
      model and proves the patient is alive, Lab 10 proves the ULD loss's five properties and
      measures the real cross-tokenizer baseline, and Lab 11's contamination checker first
      caught the chat template's scaffolding pretending to be contamination and then found
      (and remediated) real near-duplicates in my own eval split.
- [x] `code/kd_pipeline.py`: the Tier 2 engineering layer. Memory plans, bandwidth/KV
      arithmetic, the top-k cache writer/reader with corpus fingerprinting, the entropy
      monitor, the ULD loss, run manifests. Each piece is asserted by the labs' Part A cells.
- [x] **Solutions, all 13 labs.** One notebook per lab in `solutions/`, solving every
      exercise with the approach explained before the code. Exercises answerable without
      training were executed live (all of Labs 00 to 02, and the measurement/mechanism parts
      of the rest); exercises that need real training carry full working code behind the same
      `RUN_TRAINING` flag with expected ranges. A few solutions surfaced genuine findings:
      the library's `seq_kd=True` path is dead code at `lmbda=1.0` (it needs `lmbda=0.0`),
      and a couple of the exercises' premises got honestly refuted by the data rather than
      forced. Try each exercise before opening its solution; the format rules live in
      `SOLUTIONS_SPEC.md`.
- **Run order on the box:** Lab 03 Part A builds the shared corpus the later labs reuse. After
  that, labs are independent per the artifact chain in `ROADMAP.md`.
- **A note on drift:** the Tooling paragraph below describes TRL's distillation surface as I
  first found it. The installed TRL 1.9.2 no longer has `loss_top_k` or the teacher-server
  constraint, and its `DistillationTrainer` is now the always-on-policy path with no `lmbda`.
  This is why the labs introspect the installed configs at runtime instead of trusting any
  document, including this one.

---

## Sources

Each unit is written against primary sources, not recollection, and the reading happens before
the chapter gets drafted. Full link list in `REFERENCES.md`.

**Foundations.** Hinton, Vinyals, Dean (2015) on the temperature objective. Müller, Kornblith,
Hinton on why a label-smoothed teacher distills worse. Beyer et al., *A good teacher is patient
and consistent*, on distillation as function matching under identical augmentation.

**Sequence models.** Kim and Rush (2016) on sequence-level KD. MiniLLM on reverse KL with
policy-gradient variance reduction. Agarwal et al. (2306.13649) on GKD, the paper the TRL
trainer implements.

**On-policy, current.** The 2026 on-policy distillation survey and its f-divergence framing,
plus the failure-mode papers on the sampled-token estimator, entropy and length collapse,
late-token instability, off-policy cold start, and OOD-prompt mixing.

**Tooling.** TRL `DistillationTrainer` docs (`trl.experimental.distillation`) covering `lmbda`,
`beta`, `loss_top_k`, the generation buffer, the external vLLM teacher server, and the
constraint that `loss_top_k` must be exactly 1 with a remote teacher when `beta > 0` (see the
drift note above; several of these have since moved). Plus the GKD and GOLD trainer pages for
the cross-tokenizer path.

**Compression.** Minitron and Sheared-LLaMA on prune-then-distill. DistilBERT and TinyBERT on
layer selection and feature matching. DeepSeek-R1-Distill as the canonical "SFT on teacher
traces, no logits, no RL" reference point.

**Security.** T-MTB on why ordinary LLM backdoors mostly don't survive distillation, and how
composite triggers built from tokens common in distillation corpora do transfer. DistillGuard
on evaluating defenses against distillation-based extraction.

---

## Layout

```
ModelDistillation/
  README.md                 this file
  ROADMAP.md                the lab-by-lab plan for units 03-12
  REFERENCES.md             full reading list with links, organised by lab
  requirements.txt          versions the labs were verified against
  code/kd_core.py           the primitives, library-agnostic, [B,T,V] in
  code/kd_pipeline.py       the Tier 2 engineering layer
  labs/                     numbered notebooks, 00-12
  solutions/                worked exercise solutions, one per lab
  figures/                  figures written by executed labs
  data/, runs/              generated on the box; gitignored except JSON receipts
```

Run the labs with the repo root as working directory. They resolve `../code` on `sys.path`.
