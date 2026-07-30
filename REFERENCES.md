# References

The reading list for the course, organised by the lab that leans on each source. The rule from
the README applies to you as much as it did to the build: read the source *before* (or right
after) the lab that uses it — each lab tells you which claims it staked on which paper, and the
papers land differently once you have run the thing they describe.

## Foundations (Labs 00–03)

- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network* (2015).
  The temperature objective, dark knowledge, the T² factor. Lab 01 verifies its math; Lab 03
  runs it. https://arxiv.org/abs/1503.02531
- Müller, Kornblith, Hinton — *When Does Label Smoothing Help?* (2019).
  Why a label-smoothed teacher distills worse — the wrong-answer probabilities are the
  signal. Lab 03's exercise 2 reproduces the effect. https://arxiv.org/abs/1906.02629
- Beyer et al. — *Knowledge distillation: A good teacher is patient and consistent* (2021).
  Function matching, patience beating teacher size, the capacity-gap framing Lab 03's gap
  arms test. https://arxiv.org/abs/2106.05237
- Schulman — *Approximating KL Divergence* (blog).
  The k1/k2/k3 estimators Lab 00 §9 measures in both regimes. http://joschu.net/blog/kl-approx.html

## Sequence-level and on-policy (Labs 05–07)

- Kim, Rush — *Sequence-Level Knowledge Distillation* (2016).
  SeqKD: train on the teacher's mode. Lab 06's first arm. https://arxiv.org/abs/1606.07947
- Gu et al. — *MiniLLM: Knowledge Distillation of Large Language Models* (2023).
  Reverse KL for generation, policy-gradient variance reduction. Background for Lab 05's β=1
  column and Lab 07's estimator concerns. https://arxiv.org/abs/2306.08543
- Agarwal et al. — *On-Policy Distillation of Language Models (GKD)* (2023).
  The paper TRL's GKD trainer implements: `lmbda`, `beta`, exposure bias. Labs 05 and 07's
  primary source. https://arxiv.org/abs/2306.13649
- *A Survey of On-Policy Distillation for Large Language Models* (2026).
  The f-divergence organisation of the field and the failure-mode catalogue (sampled-token
  estimator, entropy/length collapse, cold start) Labs 00, 05, 07, and 11 draw on.
  https://arxiv.org/abs/2604.00626
- DeepSeek-AI — *DeepSeek-R1* (2025), distillation section.
  The canonical "SFT on teacher traces, no logits, no RL" reference — Lab 06's trace-SFT arm.
  https://arxiv.org/abs/2501.12948

## Compression and initialisation (Lab 09)

- Muralidharan et al. — *Compact Language Models via Pruning and Knowledge Distillation
  (Minitron)* (2024). Measured layer importance, prune-then-distill economics. Lab 09's
  method. https://arxiv.org/abs/2407.14679
- Xia et al. — *Sheared LLaMA* (2023). Structured pruning + staged recovery, the contrast in
  Lab 09's exercise 2. https://arxiv.org/abs/2310.06694
- Sanh et al. — *DistilBERT* (2019). Layer-selection initialisation, the lineage's start.
  https://arxiv.org/abs/1910.01108
- Jiao et al. — *TinyBERT* (2019). Feature matching with projectors — Lab 10's B·2 and Lab
  09's exercise 4. https://arxiv.org/abs/1909.10351

## Cross-tokenizer (Lab 10)

- Boizard et al. — *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation
  Loss* (2024). The sorted-probability loss `kd_pipeline.uld_sorted_loss` implements and Lab
  10 property-tests. https://arxiv.org/abs/2402.12030

## Evaluation and security (Lab 11)

- *Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation* (2025).
  Why ordinary backdoors wash out and composite triggers built from distillation-corpus
  tokens survive — Lab 11's marker-transfer protocol tests exactly this.
  https://arxiv.org/abs/2510.18541
- *DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation* (2026).
  The extraction-defense framing (logit truncation, noise, watermarking) behind Lab 11's
  exercises 3–4. https://arxiv.org/abs/2603.07835
- EleutherAI — *lm-evaluation-harness*. Lab 11's benchmark pass.
  https://github.com/EleutherAI/lm-evaluation-harness

## Tooling documentation (Labs 04, 06–08, 10)

The standing rule: these pages describe `trl.experimental`, which drifts — the labs introspect
the installed objects and treat these docs as orientation, not contract.

- TRL GKD trainer (Labs 06–07): https://huggingface.co/docs/trl/main/en/gkd_trainer
- TRL DistillationTrainer, always-on-policy + vLLM teacher serving (Labs 07–08):
  https://huggingface.co/docs/trl/main/en/distillation_trainer
- TRL GOLD trainer, cross-tokenizer (Lab 10): https://huggingface.co/docs/trl/main/en/gold_trainer
- vLLM OpenAI-compatible server (Lab 08; `prompt_logprobs` is the payload Lab 08's client
  parses): https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Transformers Trainer/callbacks (Labs 03, 07): https://huggingface.co/docs/transformers/main_classes/trainer

## Models and data used throughout

- SmolLM2 (the course's model family; the paper documents the data recipe SmolTalk comes
  from): https://arxiv.org/abs/2502.02737 — models at https://huggingface.co/HuggingFaceTB
- SmolTalk / smol-smoltalk (the shared corpus Lab 03 builds from):
  https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- Qwen2.5 (Lab 10's cross-tokenizer teacher): https://arxiv.org/abs/2412.15115

## Platform (kept out of the labs by design, useful to you)

- DGX Spark PyTorch & CUDA guide (community): https://github.com/martimramos/dgx-spark-ml-guide
- DGX Spark ML setup guide (natolambert): https://github.com/natolambert/dgx-spark-setup
- NGC PyTorch containers: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- scitrera Spark container builds: https://github.com/scitrera/cuda-containers
