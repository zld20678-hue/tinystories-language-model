# Learning Log

This is the less polished version of the project story: what I tried, what confused me, what changed my mind, and what I would do differently next time.

I started this as a non-technical founder trying to move beyond surface-level familiarity with language models. My goal was not to memorize terminology. I wanted to be able to trace the path from raw text all the way to a model generating tokens, and then reason about why an inference system was fast or slow.

## 1. First milestone: make a tiny model learn anything at all

I used TinyStories because the language is simple enough that a very small model can show visible progress quickly.

My first training subset had:

- 21,197 stories
- about 4.83M processed tokens
- 2,000 validation stories
- about 398K validation tokens

I trained a custom BPE tokenizer with a vocabulary of 4,096. This was one of the first concepts that clicked for me: the model never directly sees English words. The tokenizer turns text into token IDs, and those IDs become the model's input.

The baseline model was intentionally small:

- 4 Transformer layers
- hidden size 256
- 4 attention heads
- context length 128
- about 5.29M trainable parameters

I used AdamW with a learning rate of 3e-4 and batches of 32 × 128 tokens.

Before committing to a real run, I did a 20-step sanity test. Loss dropped from roughly 8.50 to 6.47. That was a small moment, but useful: it meant the model, loss function, backward pass, and optimizer were at least connected correctly.

After 1,000 steps, validation loss was around 3.55. The generated text looked recognizably English and had TinyStories-style dialogue and character introductions, but it would lose track of objects and characters pretty quickly.

I kept training the same 1% subset. Validation loss moved to roughly 3.13 at 2,000 steps and 2.95 at 4,000 steps. The model was still improving, but the curve was flattening.

That was the first time I had to make a real experimental decision instead of just following steps: keep repeating the same data, or change the data.

## 2. More data instead of more repetition

I kept the architecture fixed and moved from 1% to 5% of TinyStories.

The new training set contained 23,775,548 tokens. With a 32 × 128 batch, one dataset-sized token budget is about 5,805 steps, so I trained a fresh model from random initialization for 6,000 steps.

Final validation loss: **2.6264**.

The stories were noticeably cleaner. Grammar and short-range coherence improved, while long-range logic and entity consistency were still weak. One generation could keep Lily, her mom, a park, and a butterfly together for a while, then suddenly decide the butterfly was a bird.

That was enough for me to stop optimizing model quality. The project goal was not to squeeze every last point out of TinyStories; I wanted to understand the inference side too.

## 3. My first inference benchmark was wrong

The first time I timed generation, I got about 18 tokens/sec. Later runs were hundreds of tokens/sec.

The model had not magically become 10× faster. I had mixed cold-start / MPS setup effects into the measurement.

So I rebuilt the benchmark with warm-up runs and repeated measurements. The reliable single-sequence baseline was:

- median latency: 0.3001 s for 100 generated tokens
- median throughput: 333.26 tokens/sec

That was a useful lesson by itself: before optimizing something, make sure the number you are optimizing is real.

## 4. Profiling before guessing

I split the generation loop into rough components:

- Transformer forward: 75.2%
- sampling: 12.0%
- token concatenation: 6.2%
- other: 6.6%

The obvious bottleneck was the repeated model forward pass.

From there I tried several ideas, and most of them did not work.

### `torch.compile`

This sounded like the easiest win. It was not.

- baseline: 333.26 tok/s
- compiled: 316.36 tok/s
- change: -5.07%

For this small model on MPS, compilation overhead / execution behavior did not help the workload.

### Preallocating the generated-token buffer

Instead of repeatedly growing the output tensor with `torch.cat`, I allocated the space in advance.

- result: 334.65 tok/s
- improvement: about +0.42%

Technically faster, but too small to call meaningful.

## 5. KV cache: mathematically right, practically worse

KV caching looked like the most natural optimization because the baseline repeatedly recomputed attention over old tokens.

Before measuring speed, I wanted to know whether my cached decoder was actually computing the same thing as the original model.

I compared next-token logits in two places:

1. after prompt prefill
2. after one incremental decode step

The maximum absolute difference was about 1.9e-6, the mean difference was around 1e-7, and both implementations predicted the same next tokens.

So the implementation was numerically consistent enough to benchmark.

Then the surprise:

- original full-context path: 333.26 tok/s
- KV cache v1: 130.89 tok/s

The theoretically cheaper version was much slower.

I profiled the KV implementation. Incremental decode consumed 86.8% of runtime. My first cache implementation also kept growing K/V tensors using `torch.cat`, so I replaced that with preallocated K/V buffers.

That improved the KV version from 130.89 to 140.06 tok/s, which confirmed that cache allocation was part of the problem. But the result was still far behind the original model.

The conclusion I took from this was important: reducing FLOPs is not the same as reducing wall-clock time. My handwritten incremental decoder turned a few efficient, larger operations into many tiny tensor operations and GPU kernel launches. On this model size and Apple MPS backend, that overhead dominated.

## 6. Batching was the first large win

The KV-cache result made me think differently about the hardware. Instead of asking “How do I make each sequence do less work?”, I tested “Can I give the GPU more parallel work at once?”

I benchmarked 100-token generation with different batch sizes:

| Batch | Median batch latency | Aggregate throughput | Per-sequence throughput |
| ---: | ---: | ---: | ---: |
| 1 | 0.2549 s | 392.31 tok/s | 392.31 tok/s |
| 2 | 0.3155 s | 633.84 tok/s | 316.92 tok/s |
| 4 | 0.3405 s | 1,174.80 tok/s | 293.70 tok/s |
| 8 | 0.4483 s | 1,784.52 tok/s | 223.07 tok/s |
| 16 | 0.7153 s | 2,236.97 tok/s | 139.81 tok/s |
| 32 | 1.2341 s | 2,593.03 tok/s | 81.03 tok/s |

Batch 32 increased aggregate throughput from 392.31 to 2,593.03 tok/s, roughly **6.61×**.

It did not make one request 6.61× faster. In fact, per-request latency went up. The gain was system throughput: more total work completed per second.

That distinction between latency and throughput is probably the systems concept I understand most differently now than when I started.

## Where I would go next

If I continued this project, I would split the work in two directions.

For model quality, I would test more data, a larger context window, and a somewhat larger model.

For inference, I would stop extending the Python-level KV-cache implementation and instead test a decoder path designed around fused or backend-friendly incremental kernels.

The biggest lesson was not that one optimization technique is universally better. It was that profiling, hardware behavior, and workload shape can matter more than the theoretical complexity on paper.