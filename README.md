# TinyStories Language Model from Scratch

I came into this project as a non-technical founder who wanted to get much closer to the technical work instead of treating AI infrastructure as a black box. In a short window, I wanted to go from “I roughly know what a language model does” to being able to train one from scratch, debug it, profile it, explain the important code paths, and make a real inference optimization decision based on data rather than intuition.

So this repo is less about building an impressive story generator and more about the learning curve itself: tokenizer → data pipeline → Transformer → training → validation → generation → profiling → inference optimization.

The final model is small by modern standards — about 5.29M parameters on an Apple M4 using PyTorch MPS — which made it possible to inspect the whole stack instead of hiding behind a large framework.

## Model setup

- 4 Transformer layers
- hidden size: 256
- 4 attention heads
- context length: 128 tokens
- custom BPE tokenizer
- vocabulary size: 4,096
- AdamW optimizer
- batch size: 32

The main files are:

- `train_tokenizer.py` — train the BPE tokenizer
- `prepare_data.py` / `prepare_data_5pct.py` — turn stories into token IDs
- `model.py` — the Transformer
- `train.py` / `train_5pct.py` — training and validation
- `resume_train.py` / `continue_train_*.py` — continue from checkpoints
- `generate.py` — text generation
- `benchmark_*.py` — inference experiments
- `model_kv*.py` — KV-cache experiments
- `docs/learning_log.md` — notes from the process

## 1. Getting a baseline model to actually learn

I started with 21,197 TinyStories examples, roughly 1% of the training set. The first useful sanity check was a 20-step run: loss fell quickly, which at least told me that the forward pass, loss, backpropagation, and optimizer were connected correctly.

After 1,000 steps, validation loss was around **3.55**. The model had picked up basic English structure and the usual TinyStories rhythm, but the stories still lost track of people and objects pretty easily.

I then kept training the same 1% subset:

| Training setup | Validation loss |
| --- | ---: |
| 1% data, 1,000 steps | ~3.55 |
| 1% data, 2,000 steps | ~3.13 |
| 1% data, 4,000 steps | ~2.95 |

The model was still improving, but the gains were clearly slowing down. At that point, continuing to grind on the same small subset felt less interesting than changing the data.

## 2. More data was more useful than more repetition

For the next experiment I kept the model, tokenizer, context length, optimizer family, and hardware fixed. The main change was increasing TinyStories from 1% to 5%.

The 5% subset contained **23,775,548 training tokens**. At 32 × 128 tokens per step, that is about 5,805 batch-sized chunks, so I trained a fresh model for 6,000 steps.

Final validation loss: **2.6264**.

That was a much more convincing improvement than simply continuing to recycle the 1% subset. The generated stories became noticeably more grammatical and locally coherent. They were still not great at long-range logic or keeping every entity straight, but the model had reached a useful stopping point for this project.

One sample from the 5% model started:

> Once upon a time, there was a little girl named Lily. She loved to go to the park with her mommy...

It later confused a butterfly and a bird, which is a good summary of the model at this stage: sentence-level structure was much better, story-level state tracking was still weak.

I stopped scaling the model here and moved to inference. The goal was to learn something about systems behavior rather than spend the rest of the project squeezing down validation loss.

## 3. Inference: the part that surprised me

My first generation timing was misleadingly slow because it included cold-start / MPS setup effects. I rebuilt the benchmark with warm-up runs and repeated measurements.

For the 5% model, generating 100 tokens gave a reproducible single-sequence baseline of roughly:

- median latency: **0.3001 s**
- median throughput: **333.26 tokens/s**

A simple profiler split the generation loop into a few pieces:

| Component | Share of measured time |
| --- | ---: |
| Transformer forward | 75.2% |
| Sampling | 12.0% |
| Token concatenation | 6.2% |
| Other | 6.6% |

So the obvious target was the repeated model forward pass.

### Things I tried that did *not* help

This ended up being the most useful part of the project.

`torch.compile` sounded like the easiest win, but on this small MPS workload it was actually slower:

- baseline: **333.26 tok/s**
- compiled: **316.36 tok/s**
- change: **-5.07%**

I also replaced repeated output `torch.cat` calls with a preallocated token buffer. That moved throughput to **334.65 tok/s**, only about **+0.42%**. In practice, that was too small to treat as a meaningful win.

### KV cache: correct, but slower

Because full-context decoding repeatedly recomputes attention for old tokens, KV caching looked like the natural next step.

Before timing it, I checked correctness against the original model at two points:

1. after prompt prefill
2. after one incremental decode step

The maximum absolute logit difference was about **1.9e-6**, and both implementations predicted the same next tokens. So the cached path was numerically consistent before I benchmarked it.

Performance was a different story:

| Version | Median throughput |
| --- | ---: |
| Original full-context path | 333.26 tok/s |
| KV cache v1 | 130.89 tok/s |
| KV cache + preallocated K/V buffers | 140.06 tok/s |

Profiling the cached version showed that **86.8%** of its runtime was still inside incremental decoding.

Preallocating the K/V buffers did help the KV version itself (130.89 → 140.06 tok/s), which confirmed that repeated cache growth was part of the overhead. But it was nowhere near enough to beat the original Transformer path.

My takeaway was that lower theoretical FLOPs do not automatically mean lower wall-clock latency. For a small 5.3M model with a 128-token context window, the hand-written cached decoder broke the work into lots of small operations. On Apple MPS, that overhead outweighed the arithmetic I had saved.

## 4. The optimization that actually worked: batching

The KV-cache result suggested that this workload benefited more from larger parallel operations than from many tiny incremental ones, so I tested batching.

Each sequence still generated 100 new tokens. I changed only the number of sequences processed together.

| Batch | Median batch latency | Aggregate throughput | Throughput per sequence |
| ---: | ---: | ---: | ---: |
| 1 | 0.2549 s | 392.31 tok/s | 392.31 tok/s |
| 2 | 0.3155 s | 633.84 tok/s | 316.92 tok/s |
| 4 | 0.3405 s | 1,174.80 tok/s | 293.70 tok/s |
| 8 | 0.4483 s | 1,784.52 tok/s | 223.07 tok/s |
| 16 | 0.7153 s | 2,236.97 tok/s | 139.81 tok/s |
| 32 | 1.2341 s | **2,593.03 tok/s** | 81.03 tok/s |

From batch 1 to batch 32, aggregate throughput increased from **392.31 to 2,593.03 tokens/s**, about **6.61×**.

That is not a 6.61× latency improvement for one user. It is a throughput improvement: the GPU completes much more total work per second when several sequences are processed together. The tradeoff is that the batch takes longer to finish, so individual-request latency goes up.

That distinction ended up being one of the most important things I learned in this project.

## What I would do next

If I had more time, I would separate two directions rather than keep tuning the same setup:

- **model quality:** more training data, a larger context window, and a somewhat larger model
- **inference:** test a decoder implementation designed around fused / backend-friendly incremental kernels instead of a Python-level KV-cache path

The main lesson from this project was less “KV cache is good” or “batching is good” and more that **the best optimization depends on the workload and the hardware**. Profiling mattered more than guessing.