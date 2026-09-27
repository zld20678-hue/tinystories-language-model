# TinyStories Language Model from Scratch

I came into this project as a non-technical founder who had spent a lot of time around AI products without ever going all the way down to the mechanics. I wanted to change that quickly.

So instead of starting with an API or fine-tuning an existing model, I trained a small autoregressive Transformer from scratch on TinyStories and followed the whole pipeline myself: tokenizer → data preparation → model → training → validation → generation → profiling → inference optimization.

The goal was not to build an impressive story generator. A 5.29M-parameter model is tiny by modern standards. The goal was to get to the point where I could look at a training curve, a profiler output, or an inference tradeoff and explain what was happening instead of treating the model as a black box.

The most useful part was that several of my "obvious" optimization ideas did not work.

> **Best result:** batching increased aggregate inference throughput from **392.31 to 2,593.03 tokens/s** on an Apple M4 — about **6.61×** — while also making the latency/throughput tradeoff impossible to ignore.

## Project at a glance

- **Model:** 4-layer autoregressive Transformer
- **Parameters:** ~5.29M
- **Hidden size:** 256
- **Attention heads:** 4
- **Context length:** 128 tokens
- **Tokenizer:** custom BPE, vocabulary 4,096
- **Training data:** TinyStories
- **Optimizer:** AdamW
- **Hardware:** Apple M4, PyTorch MPS
- **Best validation loss:** 2.6264 on the 5% training-data experiment

## Repository map

The root files follow the order I built the system:

- `train_tokenizer.py` — trains the custom BPE tokenizer
- `prepare_data.py` / `prepare_data_5pct.py` — converts TinyStories text into token IDs
- `model.py` — defines the Transformer
- `train.py` / `train_5pct.py` — training + validation loops
- `resume_train.py` / `continue_train_*.py` — checkpoint continuation experiments
- `generate.py` — autoregressive generation
- `compare_models.py` — compares checkpoints / generations
- `benchmark_*.py` — inference benchmarks
- `profile_*.py` — bottleneck profiling
- `model_kv*.py` — custom KV-cache implementations
- `validate_kv*.py` — numerical correctness checks for the cached decoder
- `docs/learning_log.md` — the longer learning / decision log

I intentionally kept the individual experiment scripts instead of collapsing everything into one polished abstraction. For this project, the progression of experiments is part of the work.

## 1. First goal: make a model actually learn

I started with 21,197 TinyStories examples, roughly 1% of the training set. My first useful sanity check was only 20 optimization steps. The loss dropped quickly, which was enough to tell me that the forward pass, loss, backward pass, and optimizer were connected correctly.

After 1,000 steps, validation loss was around **3.55**. The model had learned basic English structure and the familiar TinyStories rhythm, but it still lost track of people and objects easily.

I kept training the same 1% subset:

| Training setup | Validation loss |
| --- | ---: |
| 1% data, 1,000 steps | ~3.55 |
| 1% data, 2,000 steps | ~3.13 |
| 1% data, 4,000 steps | ~2.95 |

The model was still improving, but the gains were slowing down. That created the first decision point: keep showing the model the same small slice of data, or give it more variety?

## 2. More data beat more repetition

For the next experiment I kept the architecture, tokenizer, context length, optimizer family, and hardware fixed. The main change was increasing the TinyStories subset from 1% to 5%.

The 5% subset contained **23,775,548 training tokens**. With batches of 32 × 128 tokens, that is roughly 5,805 batch-sized chunks, so I trained a fresh model for 6,000 steps.

Final validation loss: **2.6264**.

That improvement was much more convincing than simply continuing to recycle the 1% subset. The generated stories became noticeably more grammatical and locally coherent. They were still weak at long-range logic and entity tracking, but at that point the model was good enough for the real question I wanted to explore next: what actually makes inference faster?

One sample started:

> Once upon a time, there was a little girl named Lily. She loved to go to the park with her mommy...

A few sentences later it confused a butterfly and a bird. That was a pretty accurate summary of the model: decent sentence structure, shaky world state.

## 3. Profiling changed the direction of the project

My first generation timing was misleadingly slow because it included cold-start / MPS setup effects. I rebuilt the benchmark with warm-up runs and repeated measurements.

For 100 generated tokens, the reproducible single-sequence baseline was roughly:

- median latency: **0.3001 s**
- median throughput: **333.26 tokens/s**

Then I profiled the generation loop:

| Component | Share of measured time |
| --- | ---: |
| Transformer forward | 75.2% |
| Sampling | 12.0% |
| Token concatenation | 6.2% |
| Other | 6.6% |

The obvious conclusion was that the repeated model forward pass was the place to attack.

### The easy ideas mostly failed

`torch.compile` looked like the obvious first optimization. On this workload, it was slower:

- baseline: **333.26 tok/s**
- compiled: **316.36 tok/s**
- change: **-5.07%**

I also replaced repeated output `torch.cat` calls with a preallocated token buffer. Throughput moved to **334.65 tok/s**, only about **+0.42%** — effectively noise at this scale.

Those two results were useful because they forced me to stop assuming that an optimization was helpful just because it sounded lower-level.

## 4. KV cache: mathematically right, practically slower

Full-context autoregressive decoding repeatedly recomputes attention for earlier tokens, so KV caching looked like the natural next step.

Before timing it, I first checked that my cached implementation was actually doing the same computation as the original model. I compared next-token logits at two points:

1. after prompt prefill
2. after one incremental decode step

The maximum absolute logit difference was about **1.9e-6**, and both implementations selected the same next tokens. That gave me a correctness check before performance benchmarking.

Then came the surprising part:

| Version | Median throughput |
| --- | ---: |
| Original full-context path | 333.26 tok/s |
| KV cache v1 | 130.89 tok/s |
| KV cache + preallocated K/V buffers | 140.06 tok/s |

The cache reduced repeated arithmetic, but the custom incremental decoder broke the work into many small tensor operations and GPU kernel launches. On this small 5.3M model with only a 128-token context window, that overhead dominated.

Profiling the cached version confirmed it: **86.8%** of runtime was still spent in incremental decoding.

Preallocating K/V buffers improved the KV implementation itself from 130.89 to 140.06 tok/s, so repeated cache growth was genuinely part of the problem. It just was not the main problem.

This was probably the most useful technical lesson in the project: **fewer FLOPs do not automatically mean lower wall-clock latency.**

## 5. The optimization that actually worked: batching

The KV-cache result suggested that this MPS workload liked larger parallel operations more than many tiny incremental ones. So instead of trying to reduce the work per sequence, I tried processing more sequences at once.

Each sequence still generated 100 new tokens. Only batch size changed.

| Batch | Median batch latency | Aggregate throughput | Throughput per sequence |
| ---: | ---: | ---: | ---: |
| 1 | 0.2549 s | 392.31 tok/s | 392.31 tok/s |
| 2 | 0.3155 s | 633.84 tok/s | 316.92 tok/s |
| 4 | 0.3405 s | 1,174.80 tok/s | 293.70 tok/s |
| 8 | 0.4483 s | 1,784.52 tok/s | 223.07 tok/s |
| 16 | 0.7153 s | 2,236.97 tok/s | 139.81 tok/s |
| 32 | 1.2341 s | **2,593.03 tok/s** | 81.03 tok/s |

From batch 1 to batch 32, aggregate throughput increased from **392.31 to 2,593.03 tokens/s**, roughly **6.61×**.

That is **not** a 6.61× latency improvement for one user. It is a system-throughput improvement: the GPU finishes much more total work per second when multiple sequences are grouped together. Individual-request latency gets worse.

Before this project, I would have loosely described both things as "making inference faster." I would not anymore.

## How to reproduce the project

The repository intentionally does not include downloaded datasets, `.npy` token arrays, virtual environments, or model checkpoints.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the tokenizer
python train_tokenizer.py

# 3. Prepare tokenized data
python prepare_data.py
# or the larger experiment
python prepare_data_5pct.py

# 4. Train
python train.py
# or
python train_5pct.py

# 5. Generate after placing / producing the expected checkpoint
python generate.py

# 6. Run the main inference experiments
python benchmark_inference.py
python profile_inference.py
python benchmark_batching.py
```

The exact data directories and checkpoint names in the scripts reflect the runs used for this project. Reproducing a particular experiment may require matching those paths or editing the corresponding constants.

## Reading this repo in 5 minutes

If you are reviewing the project rather than reproducing it, I would read it in this order:

1. this README for the decisions and results
2. `model.py` for the model itself
3. `train_5pct.py` for the final training loop
4. `profile_inference.py` for how I located the inference bottleneck
5. `validate_kv.py` + `model_kv.py` for the correctness-first KV-cache experiment
6. `benchmark_batching.py` for the final throughput result
7. `docs/learning_log.md` for the longer learning process

## What I would do next

If I continued this project, I would split the next work into two tracks:

- **model quality:** more data, a larger context window, and a somewhat larger model
- **inference systems:** a decoder designed around fused / backend-friendly incremental kernels rather than a Python-level KV-cache path

The part I care about most is not that batching "won." It is that profiling changed my assumptions several times. Going from product-level intuition to being able to reason about loss curves, numerical correctness, kernel overhead, and latency-vs-throughput tradeoffs was the actual point of the exercise.