# Distillation of Generative Models

A progressive lab course, from the objective function to a reproducible research pipeline, with
the real-model track sized for a single workstation with **128 GB of unified CPU–GPU memory**
at ~273 GB/s of bandwidth (arm64, CUDA compute capability sm_121).

---

## How this course is built, and why it matters

Labs come in two tiers, and each one says which it is on the first line.

**Tier 1 — executed and asserted.** Runs on any machine in seconds. Operates on synthetic logit
tensors, real tokenizers, and cached artifacts. Every claim is checked with an `assert` against
autograd or a closed form. These labs were executed end to end during the build; the outputs you
see were produced by running them.

**Tier 2 — grounded and unexecuted.** Real distillation runs on real checkpoints. Written against
cited library APIs with parameters taken from official documentation and published recipes, sized
for the memory and bandwidth budget above. **These were not executed during the build.** A real
training run cannot be validated on build hardware, and a green checkmark produced by a toy
stand-in would be
a false assurance about your box. Each Tier 2 lab instead carries expected output ranges, the
failure signatures to watch for, and a verification checklist you run first.

The split follows a rule worth internalising: **the things worth executing are the things that
need no training, and the things that need training cannot be validated anywhere but your own
hardware.** The objective, the masking, the caching arithmetic, the tokenizer alignment — these
are exact, fast to check, and where silent bugs actually live. A wrong `beta` or an unshifted
mask trains the wrong thing without raising an exception or bending the loss curve.

That is not hypothetical. During the build of this course, `kd_core.gjsd` shipped with a docstring
that described `beta` backwards relative to TRL's convention. The *code* was correct; the
*documentation* was inverted, which is worse, because a reader trusts the comment. It was caught
by reading the current TRL docs and settled in one second by evaluating four numbers. Lab 01
contains that check as a permanent assertion.

---

## Reading the hardware before designing the pipeline

Two measured facts drive every Tier 2 decision.

**Capacity is generous, bandwidth is not.** 128 GB of coherent unified memory means a 70B teacher
*fits*. 273 GB/s shared across CPU and GPU means it will not be *fast*. LMSYS measured GPT-OSS 20B
in MXFP4 at roughly 2,053 tok/s prefill against 49.7 tok/s decode on this hardware class. That
~40:1 ratio is the number to design around.

**Therefore: sort every workload by who is decoding.**

| Workload | Shape | Who decodes | Cost on this box |
|---|---|---|---|
| Caching teacher top-k logits over a corpus | prefill | nobody | cheap, pay once |
| Teacher scoring student rollouts | prefill | nobody | cheap |
| Student rollouts for on-policy distillation | decode | **the student** (small) | moderate |
| Teacher generating a SeqKD corpus | decode | **the teacher** (large) | expensive |

I want to flag a correction here, because I got this wrong earlier while writing this course and
the reasoning is instructive. Seeing 40:1, my first conclusion was "on-policy distillation is the
expensive path on a bandwidth-pinned box, so do a long off-policy phase and a short on-policy
phase." That is wrong, and the error was forgetting *which model generates*. In on-policy
distillation the rollouts come from the **student**, which is the small model; the teacher only
ever does a scoring forward pass. A 1.7B bf16 student is ~3.4 GB, so weight reads alone cap it
near 273/3.4 ≈ 80 decode steps/sec, and batching amortises that across the whole batch.

The genuinely expensive pattern is the opposite one: **sequence-level KD, where the large teacher
generates the corpus.** A 32B bf16 teacher is 64 GB, capping decode near 4 steps/sec before
anything else binds.

So on this hardware class the guidance inverts the usual advice about which technique is affordable:

- Prefer on-policy distillation and cached-logit off-policy training, both of which lean on
  prefill and small-model decode.
- Treat teacher-generated corpora as a *purchased asset*. Generate once and cache, use an existing
  published teacher-generation dataset, or rent an hour of H100 for that phase specifically.
- Never pay for large-teacher decode twice.

### Memory budget

Full fine-tuning costs about **16 bytes per parameter** (bf16 weights and grads, fp32 Adam moments
and master weights). LoRA collapses the optimizer share to near nothing.

| Configuration | Teacher (bf16 inference) | Student | Total | Verdict |
|---|---|---|---|---|
| 8B teacher + 1.7B student, full FT | 16 GB | 27 GB | ~43 GB | comfortable, good default |
| 14B teacher + 1.7B student, full FT | 28 GB | 27 GB | ~55 GB | comfortable |
| 8B teacher + 4B student, full FT | 16 GB | 64 GB | ~80 GB | workable, watch KV cache |
| 32B teacher + 8B student, LoRA | 64 GB | ~18 GB | ~82 GB | viable, quantize teacher if tight |
| 32B teacher + 4B student, full FT | 64 GB | 64 GB | ~128 GB | will not fit, quantize or offload |

Leave real headroom for KV cache and activations; the table is weights and optimizer state only.

### arm64 notes

Wheel availability is the recurring friction on sm_121/aarch64. Reuse a known-good vLLM container
for the teacher server rather than building fresh, and verify `flash-attn`, `bitsandbytes`, and
Liger availability before designing a run around them. TRL's teacher-server path explicitly does
not support Liger kernels.

---

## Course spine

Prerequisite refreshers assume you can read PyTorch and have trained a transformer, and no more.

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

- [x] `code/kd_core.py` — divergences, top-k caching, alignment, diagnostics. Verified.
- [x] **Lab 00** — executed, 10/10 code cells, all assertions passing. Covers softmax/logsumexp
      numerics, the H/CE/KL identity, KL's non-metric behaviour (and that sqrt-JSD *is* a
      metric), the f-divergence family verified generator-vs-direct, Pinsker and boundedness,
      bf16/fp16/fp32 loss-computation policy, the whiteboard gradient, and the k1/k2/k3 sampled
      KL estimators — including the measured result that k3's variance advantage holds only when
      student ≈ teacher, which is Unit 07's cold-start problem in miniature.
- [x] **Lab 01** — executed, 12/12 cells, all assertions passing.
- [x] **Lab 02** — executed, 7/7 code cells, all assertions passing. Real tokenizers (GPT-2,
      Qwen2.5, SmolLM2) and a real teacher/student pair (SmolLM2 360M → 135M). Centerpiece:
      `kd_core`'s external shift+mask reproduces HF's internal `model(labels=...).loss` to float
      precision on a ragged chat-templated batch. Also measured on real logits: top-k truncation
      bias runs in *opposite directions* for renormalise (overstates) vs tail-bucket
      (understates) — pick `k` by measurement, not doctrine — plus cross-tokenizer boundary
      overlap and cache/prefill pricing at measured throughput.
- [x] **Labs 03–12 — built.** Every Tier 2 lab follows the three-part shape from `ROADMAP.md`:
      Part A (pre-flight, executed and asserted during the build — config discipline, memory
      plans, loss/format/detector verification), Part B (the training run, gated behind a
      `RUN_TRAINING`-style flag, **not executed during the build**, grounded against the
      installed library APIs), Part C (expected ranges, failure signatures, the verdict to
      write, exercises). Highlights that executed for real during the build: Lab 04's cache
      format survives a tamper attack; Lab 08's remote-scoring client verifies end-to-end
      against a live mock server; Lab 09 performs real depth-pruning surgery on a 135M model
      and proves the patient alive; Lab 10 proves the ULD loss's five properties and measures
      the real cross-tokenizer baseline; Lab 11's contamination checker caught the chat
      template's scaffold masquerading as contamination and then found (and remediated) real
      near-duplicates in the course's own eval split.
- [x] `code/kd_pipeline.py` — the Tier 2 engineering layer: memory plans, bandwidth/KV
      arithmetic, the top-k cache writer/reader with corpus fingerprinting, the entropy
      monitor, the ULD loss, run manifests. Every piece asserted by the labs' Part A cells.
- **Run order on the training box:** Lab 03 Part A builds the shared corpus the later labs
  reuse; after that, labs are independent per their `ROADMAP.md` artifact chain.
- **A note on drift:** the Tooling paragraph below describes TRL's distillation surface as it
  was first grounded; the build later found `loss_top_k` and the teacher-server constraint
  absent from the installed TRL 1.9.2, whose `DistillationTrainer` is now the always-on-policy
  path (no `lmbda`). The labs therefore introspect the installed configs at runtime instead of
  trusting any document, this one included.

---

## Sources

Each unit is written against primary sources rather than recollection, and the fetch happens
before the chapter is drafted.

**Foundations.** Hinton, Vinyals, Dean (2015) on the temperature objective. Müller, Kornblith,
Hinton on why a label-smoothed teacher distills worse. Beyer et al., *A good teacher is patient
and consistent*, on distillation as function matching under identical augmentation.

**Sequence models.** Kim and Rush (2016) on sequence-level KD. MiniLLM on reverse KL with
policy-gradient variance reduction. Agarwal et al. (2306.13649) on GKD, which is the paper the
TRL trainer implements.

**On-policy, current.** The 2026 on-policy distillation survey and its f-divergence organisation;
the failure-mode papers on the sampled-token estimator, entropy and length collapse, late-token
instability, off-policy cold start, and OOD-prompt mixing.

**Tooling.** TRL `DistillationTrainer` docs (`trl.experimental.distillation`), covering `lmbda`,
`beta`, `loss_top_k`, the generation buffer, the external vLLM teacher server, and the constraint
that `loss_top_k` must be exactly 1 with a remote teacher when `beta > 0`. Plus the GKD and GOLD
trainer pages for the cross-tokenizer path.

**Compression.** Minitron and Sheared-LLaMA on prune-then-distill. DistilBERT and TinyBERT on
layer selection and feature matching. DeepSeek-R1-Distill as the canonical "SFT on teacher traces,
no logits, no RL" reference point.

**Security.** T-MTB on why ordinary LLM backdoors mostly fail to survive distillation, and how
composite triggers built from tokens common in distillation corpora do transfer. DistillGuard on
evaluating defenses against distillation-based extraction.

---

## Layout

```
generative-model-distillation/
  README.md                 this file
  code/kd_core.py           the primitives, library-agnostic, [B,T,V] in
  labs/                     numbered notebooks
  figures/                  figures written by executed labs
```

Run Tier 1 labs with the repo root as working directory; they resolve `../code` on `sys.path`.
