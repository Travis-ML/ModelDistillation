# Solutions-notebook spec (addendum to STYLE_SPEC.md)

Every rule in STYLE_SPEC.md applies to all solution prose: plain language, terms defined at
first use, no em-dashes, no mention of specific hardware vendors or product names, first
person where natural, numbers carry one-line derivations.

## Naming and layout

One solutions notebook per lab, built by a builder script (same pattern as the labs):

- builder: `build_solNN.py` in the repo root
- output: `solutions/sol-NN-<same-slug-as-the-lab>.ipynb`
- notebooks run with `solutions/` as working directory; they resolve `../code` on `sys.path`
  and reference shared data as `../data/...` (same relative depth as `labs/`)

## Per-exercise structure (strict)

For each exercise in the lab's Exercises section, in order:

1. **Markdown: the exercise, restated.** Quote or faithfully paraphrase the exercise so the
   solutions notebook stands alone. Then explain the *approach* before any code: what
   question the exercise is really asking, what the plan is, and why that plan answers it.
   This explanation is the deliverable; the code is its receipt.
2. **Code: the solution.** Executed live where feasible (see budget below). Where the
   exercise requires training or serving, follow the course's Tier 2 pattern: a
   `RUN_TRAINING = False` flag set once at the top of the notebook, the full working code
   written and gated behind it, and the executable part (setup, arithmetic, measurement on
   tiny models, verification of the method on synthetic data) run live.
3. **Markdown: interpretation.** What the live output shows, or (for gated runs) the
   expected result ranges, what would count as confirming vs refuting the exercise's
   premise, and the failure signatures to watch for. Never leave a number uninterpreted.

Where an exercise is open-ended ("write the verdict", "which would you choose"), the solution
gives a worked example answer and labels it as one defensible answer, not the answer.

## Execution budget (hard rules)

- Every live cell completes in under 5 minutes on CPU.
- Live cells may load models up to SmolLM2-360M in fp32 and tokenizers freely (all are in
  the local HF cache). Nothing at or above 1.7B is loaded live.
- No live cell runs more than ~50 optimizer steps, and only on toy tensors or heads, never
  full-model fine-tuning. Real training goes behind the flag with expected results.
- Live cells must be deterministic (seeded) and end with at least one `assert` or a printed
  check that the interpretation cell references.
- Network access exists at build time; models/datasets already in the cache are preferred.

## Honesty rules

- A gated (unexecuted) solution must say so in its interpretation cell, in one line, and
  give expected ranges grounded in the parent lab's Part C.
- Do not fabricate numeric "results" for anything that did not execute. Expected ranges are
  fine; invented outputs are not.
- If an exercise's premise turns out wrong when solved (it happens; this course keeps such
  discoveries), say so plainly and show the evidence. That is a better solution than forcing
  the expected answer.

## Header cell

Each solutions notebook opens with a short markdown cell: which lab it solves, the
tier/execution status of its solutions (how many live, how many gated), and a reminder that
the reader should attempt the exercises before opening this file.
