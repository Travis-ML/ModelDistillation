# Plain-language rewrite spec for the lab notebooks

## The problem being fixed

The markdown prose in the labs was written above its own declared prerequisite. The README
promises: "you can read PyTorch and have trained a transformer, and no more." Concretely, the
reader has completed something like the Coursera Deep Learning Specialization and has trained
transformer models. That reader knows: backprop, gradients, optimizers, loss functions,
softmax and cross-entropy at the "what it computes" level, attention at the architecture
level, batching, overfitting, train/eval splits, basic PyTorch tensor code.

That reader does NOT know, and the prose must never assume:
- floating-point internals (subnormals, representable ranges, machine epsilon, flush-to-zero)
- information theory beyond "cross-entropy is the training loss" (nats, KL intuitions,
  calibration, entropy as a monitored quantity)
- systems/serving concepts (prefill vs decode, KV cache, continuous batching, rooflines,
  memory bandwidth as a bottleneck, quantization formats)
- RL concepts (on-policy/off-policy, rollouts, exposure bias)
- ML-practitioner folklore and idioms ("dark knowledge", "mode seeking", "the loss curve
  bends", "materialises a tensor", "teacher-forced")
- fp16/bf16 trade-offs, why mixed precision exists

## The rules

1. **Define every term of art at first use, in the same cell.** One clause or one sentence is
   usually enough: "prefill (scoring a prompt the model already has, which processes all
   tokens in parallel) versus decode (generating new tokens one at a time)". If a term was
   defined in an earlier lab, give a one-clause reminder on its first use in the current lab
   anyway; readers put these down for a week between labs.

2. **Expand compressed claims.** Any sentence that requires two or more inferential leaps gets
   unpacked into explicit steps, using "which means" / "so" / "because" chains. It is always
   acceptable for the prose to get longer. Plain language is not less content; it is more
   scaffolding around the same content.

3. **Numbers keep their derivations.** Never present a number like "~88 overflows fp32"
   without the one-line reason (88.7 is ln of fp32's maximum value, 3.4e38). If the
   derivation is in a nearby code cell, say "the cell below shows why".

4. **Code cells are untouchable.** Do not change a single character inside any code() cell.
   The assertions, printed strings, and numbers in them are load-bearing. Prose may be
   reworded freely as long as every technical claim it makes stays exactly as true as before.

5. **No em-dashes.** Use commas, colons, periods, or parentheses instead.

6. **Keep the voice.** First person where it already exists, direct, a practitioner explaining
   to a colleague. Keep the personal corrections ("I got this wrong at first..."). Keep the
   tier declaration on the first line of each lab. Do not add cheerleading, do not add
   "simply" or "just", do not turn it into a textbook voice.

7. **Keep the structure.** Same sections, same headers (headers may be reworded for clarity),
   same tables, same exercises (exercise wording may be clarified under the same rules).
   Every markdown cell in the original maps to exactly one markdown cell in the rewrite.

8. **Preserve all cross-references** (to other labs, sections, papers, files) and all
   technical facts, hyperparameters, model names, and expected ranges exactly.

## The exemplar

Original (from Lab 00 §2), the density this spec exists to remove:

> The practical rule this section verifies: **never take `log` of a probability you got from
> `softmax`; get log-probabilities directly from `log_softmax`.** The composed form
> underflows — fp32 runs out of subnormals near `1e-45`, so `exp(-110)` is exactly `0.0` and
> `log(0)` is `-inf` — while `log_softmax` computes the same quantity as a *difference of
> logits* and never materialises the tiny probability at all. A `-inf` log-probability
> entering a KL turns the loss into `inf` or `nan`, and in a training loop that reads as
> "divergence spike" when it is actually "wrote the formula wrong".

Rewritten to spec:

> The practical rule this section verifies: **never take `log` of a probability you got from
> `softmax`; get log-probabilities directly from `log_softmax`.** Here is why the two paths
> differ. A computer stores numbers in a format (fp32) with a smallest representable positive
> value, roughly 1e-45. Any calculation whose true answer is smaller than that gets stored as
> exactly 0.0; this is called underflow. Path one: `softmax` computes actual probabilities as
> stored numbers. A very unlikely token's true probability might be exp(-110), which is about
> 1e-48. That is below the floor, so it is stored as 0.0, and then `log(0.0)` returns
> negative infinity. Path two: `log_softmax` computes the same quantity through the identity
> log p_i = z_i - logsumexp(z), which only ever subtracts ordinary-sized numbers (logits like
> -110 and 5) from each other. The tiny probability never gets computed or stored at any
> point, so nothing underflows, and the result is the correct answer, -110, as a perfectly
> normal float. Same math, different order of operations, and only one order survives the
> number format. The last thing to know is what the bug looks like from the outside: if that
> -inf reaches a KL computation, the loss becomes inf or NaN, and on a training dashboard
> that looks like "training became unstable". You would waste hours tuning learning rates
> when the actual problem is one wrong function call.

Every markdown cell should read like the rewrite: same claims, same rigor, zero assumed
trivia, every leap walked.
