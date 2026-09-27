# Learning Log

This is the less polished version of the project story: what I was trying to understand, what I changed, what surprised me, and where I changed direction.

I started this as a non-technical founder who wanted to stop treating model training and inference as a black box. I was not trying to become an ML researcher in a weekend. I wanted enough hands-on depth to reason about the system instead of only the product around it.

## Starting from zero

The first few things I had to make concrete were embarrassingly basic but important:

- a token is not necessarily a word
- `d_model=256` is the width of each token's internal representation, not the number of Transformer layers
- model parameters are learned numbers, separate from the tokens themselves
- training changes those parameters; inference only uses them
- loss is not supposed to reach zero before a model is "done"

Once those ideas clicked, the code stopped looking like a wall of library calls and started looking like a pipeline.

## Building the first version

I trained a custom BPE tokenizer with a 4,096-token vocabulary, encoded TinyStories into token IDs, and built a 4-layer Transformer with:

- hidden size 256
- 4 attention heads
- 128-token context
- feed-forward width 4 × hidden size
- AdamW optimizer

The model ended up at about 5.29M parameters.

My first training run was deliberately tiny: 20 steps. I was not looking for model quality. I wanted a sanity check that the pipeline was alive.

The loss moved from roughly 8.5 into the 6s very quickly. That told me the model, loss function, backward pass, and optimizer were actually connected.

From there I trained the 1% TinyStories setup longer. Validation loss fell to roughly:

- ~3.55 at 1,000 steps
- ~3.13 at 2,000 steps
- ~2.95 at 4,000 steps

The stories improved, but less and less with each extra block of training. They sounded more like English, but they still mixed up objects, characters, and causal relationships.

## The first real decision: more steps or more data?

At 4,000 steps, I could keep pushing the same 1% subset or change the data distribution the model was seeing.

I chose more data.

I encoded a 5% TinyStories subset: 23,775,548 training tokens. I trained a fresh model for 6,000 steps while keeping the main architecture fixed.

Final validation loss: **2.6264**.

The improvement was easier to see in generation too. Grammar and local sentence flow got better. Long-range logic was still weak, but the model no longer felt like it was only memorizing the surface rhythm of a very small dataset.

That was a useful point to stop chasing model quality. The original task also included inference optimization, and I wanted to learn something there instead of spending all remaining time on lower validation loss.

## Inference: my first benchmark was wrong

The first time I measured generation, I got about 18 tokens/second. That number looked terrible.

It was also misleading.

The timing included cold-start and MPS initialization effects. After adding warm-up runs and repeated measurements, the real single-sequence baseline was around:

- median latency: 0.3001 s for 100 generated tokens
- median throughput: 333.26 tokens/s

That was my first good reminder that measurement is part of the engineering problem. A number is not useful just because a timer printed it.

## Profiling before optimizing

I split generation time into model forward, sampling, token concatenation, and everything else.

The result:

- Transformer forward: 75.2%
- sampling: 12.0%
- token concatenation: 6.2%
- other: 6.6%

That made the forward pass the obvious target.

### `torch.compile`: slower

I expected compilation to be the easy win.

It was not.

- baseline: 333.26 tok/s
- compiled: 316.36 tok/s
- change: -5.07%

On this small Apple MPS workload, compile overhead / execution behavior did not pay off.

### Preallocating the output token buffer: basically irrelevant

I removed repeated output `torch.cat` growth by preallocating space for generated token IDs.

Result: 334.65 tok/s, or roughly +0.42%.

That was too small to call meaningful. The profiler had already hinted at this: concatenation was only a small fraction of runtime.

## KV cache: where theory met the backend

Autoregressive decoding keeps reprocessing earlier context, so KV caching seemed like the most technically interesting optimization.

Before caring about speed, I wanted to know if my implementation was actually correct.

I compared the cached decoder's next-token logits against the original full-context model after prompt prefill and again after one incremental decoding step.

The maximum absolute difference was about 1.9e-6, and the predicted next tokens matched.

That gave me confidence that I was comparing two numerically equivalent paths rather than a fast wrong model and a slow correct one.

Then I benchmarked it.

- original full-context model: 333.26 tok/s
- KV cache v1: 130.89 tok/s

That was dramatically worse.

I profiled the KV path and found that incremental decoding itself consumed 86.8% of runtime. The custom decoder had replaced larger optimized Transformer operations with many small projections, reshapes, matrix multiplications, softmax calls, and cache updates.

I suspected repeated K/V `torch.cat` growth was adding unnecessary copies, so I preallocated the K/V cache.

That improved the KV version from 130.89 to 140.06 tok/s. So the diagnosis was partly right — cache allocation did matter — but it was nowhere near enough to beat the original path.

This was the point where I stopped trying to "make KV cache win." The more important result was understanding why reducing theoretical compute could still make the program slower on this particular hardware and workload.

## Batching: the first large positive result

The KV experiments suggested that Apple MPS was happier with larger parallel operations than with many tiny incremental ones.

So I changed the question.

Instead of asking, "How do I make one sequence use less computation?" I asked, "How much more total work can the GPU finish if I give it several sequences at once?"

Results for 100 generated tokens per sequence:

| Batch | Median latency | Aggregate throughput | Per-sequence throughput |
| ---: | ---: | ---: | ---: |
| 1 | 0.2549 s | 392.31 tok/s | 392.31 tok/s |
| 2 | 0.3155 s | 633.84 tok/s | 316.92 tok/s |
| 4 | 0.3405 s | 1,174.80 tok/s | 293.70 tok/s |
| 8 | 0.4483 s | 1,784.52 tok/s | 223.07 tok/s |
| 16 | 0.7153 s | 2,236.97 tok/s | 139.81 tok/s |
| 32 | 1.2341 s | 2,593.03 tok/s | 81.03 tok/s |

Batch 32 delivered about **6.61×** the aggregate throughput of batch 1.

But it also made each batch take much longer. That distinction — latency versus throughput — is something I had previously understood only at a vague product level. Seeing it happen directly on my own model made it much more concrete.

## What I would carry into another project

A few habits mattered more than any specific optimization:

1. **Sanity-check before scaling.** A 20-step training run can catch basic pipeline problems before hours of training.
2. **Use validation loss, not just training loss.** The question is whether the model generalizes, not whether it memorizes sampled batches.
3. **Measure before optimizing.** My first timing number was wrong, and some of my most intuitive optimizations were useless.
4. **Validate correctness before benchmarking an optimization.** The KV-cache logit comparison was more important than whether the generated story "looked okay."
5. **Do not confuse theoretical compute with actual runtime.** Backend behavior, kernel size, memory movement, and hardware utilization matter.
6. **Latency and throughput are different product decisions.** Batching improved one while hurting the other.

The project did not turn me into an ML engineer, which was never the point. It did make the stack much less mysterious — enough that I can now ask better technical questions, read the measurements, and understand why a systems choice might work on one workload and fail on another.