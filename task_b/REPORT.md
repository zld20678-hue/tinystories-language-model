# Task B — Does Context-Length Curriculum Change How a Small Language Model Learns?

## Research question

By the end of Task A, I felt comfortable with the mechanics of training a small language model. What I still did not have much intuition for was whether the *order* in which a model sees different context lengths changes how it learns.

That led to the question I wanted to test:

> **Does gradually increasing context length during training help a small Transformer learn more efficiently, or reach better final validation performance, than training with the full context length from the beginning?**

I compared two schedules:

- **Baseline:** context length 128 for the entire run
- **Curriculum:** context length 32 → 64 → 128

My initial hypothesis was not simply that curriculum would win. I expected shorter contexts to make early optimization easier, but I was unsure whether that would survive once both models had enough time at the full 128-token context. My working hypothesis was that curriculum might change the **learning trajectory** more than the **final endpoint**.

## Experimental design

I reused the same model from Task A: a 4-layer autoregressive Transformer with about 5.29M parameters, hidden size 256, 4 attention heads, a 4,096-token BPE vocabulary, and maximum context length 128. Training used TinyStories, AdamW with learning rate `3e-4`, and Apple M4 / PyTorch MPS.

For every paired comparison I held model architecture, tokenizer, dataset, optimizer, learning rate, hardware, and random seed fixed.

The full runs were also matched on:

- **24,576,000 total training tokens**
- **6,000 optimizer updates**
- **4,096 effective tokens per optimizer update**

The schedules were:

```text
Baseline:   128 × 6000 updates
Curriculum:  32 × 2000 updates
             64 × 2000 updates
            128 × 2000 updates
```

Because shorter contexts contain fewer tokens per microbatch, I used gradient accumulation to keep each optimizer update at the same effective 4,096-token budget: four microbatches at context 32, two at context 64, and one at context 128.

The final experiment was repeated with seeds `42`, `7`, and `123`.

## The first pilot looked better than it deserved

The first pilot was probably the most useful mistake in Task B.

I had matched total training tokens, and the curriculum model looked substantially better. But I later noticed that I had not matched optimizer-update count. The baseline had received 600 updates while the curriculum had received 1,400. Shorter sequences meant fewer tokens per step, so the curriculum condition had many more chances to change its parameters.

That meant the result was confounded. I could not tell whether shorter contexts helped, or whether the model simply benefited from extra optimizer updates.

I redesigned the experiment with gradient accumulation so both total tokens **and** update count matched. The corrected effect was smaller. I trusted it much more.

## Results

### Full-context validation dynamics

![Full-context validation dynamics](figures/val128_vs_tokens.png)

The baseline improved steadily on 128-token validation from the beginning. The curriculum model followed a very different path: while training on 32-token contexts, its 128-token validation loss stayed well behind baseline; after switching to 64 it started to catch up; after switching to 128 it improved rapidly and closed almost the entire gap.

This was the clearest result in the experiment: **short-context training did not teach long-context behavior for free.** The model still needed direct exposure to longer sequences. What surprised me was how quickly it adapted once that exposure arrived.

### Final validation loss across three seeds

![Final validation loss across three seeds](figures/final_loss_by_context.png)

| Validation context | Baseline mean | Curriculum mean | Mean improvement |
|---|---:|---:|---:|
| 32 | 2.8216 | 2.7825 | 0.0391 |
| 64 | 2.7354 | 2.7093 | 0.0262 |
| 128 | 2.6294 | 2.6227 | 0.0067 |

Across all three seeds, curriculum finished with lower mean validation loss at every evaluation context. The more interesting pattern was the size of the effect: it was largest at 32 tokens, smaller at 64, and very small at 128.

So although curriculum technically finished better at the full context, I would not describe this as a large final-performance gain. The average 128-token improvement was only `0.0067`.

### Transfer across curriculum stages

![Transfer across context lengths](figures/curriculum_context_transfer.png)

During the 32-token stage, validation at context 32 improved fastest. When training moved to 64, the 64-token curve improved sharply. When training finally moved to 128, the same thing happened again to the 128-token validation curve.

I would not claim this proves a specific internal mechanism, but the pattern is consistent with a simple interpretation: **the curriculum changed what the model learned first.** It learned local structure first, then extended that ability as longer dependencies were introduced.

## Interpretation

My hypothesis was partly supported.

Curriculum clearly changed the learning trajectory, and it produced a consistent final advantage at shorter contexts across all three seeds. What I did not find was a large improvement at the final 128-token context.

The three-seed mean moved from:

```text
Baseline:   2.6294
Curriculum: 2.6227
```

The result I am most comfortable defending is therefore not “context curriculum makes the model much better.” It is:

> **Context-length curriculum changed the path of learning much more than it changed the final long-context endpoint.**

If I had looked only at the final 128-token loss, I might have concluded that curriculum barely mattered. If I had looked only at the early curves, I might have concluded that it was much worse. The full trajectory told a more interesting story than either snapshot.

## Limitations

This is a deliberately small experiment. The model has about 5.29M parameters and the maximum context length is only 128 tokens, so I would not assume the same behavior transfers directly to modern long-context language models.

I also tested only one schedule (`32 → 64 → 128`) and only three random seeds. The direction of the effect was consistent, but three seeds are not enough to estimate the small 128-token effect very precisely.

Finally, the experiment shows *that* curriculum changes the training path, but not exactly *why*. Possible mechanisms include easier early optimization, different gradient statistics, and a different distribution of positional relationships. Separating those explanations would require another experiment.

## Conclusion

I started Task B asking whether a small model should learn long contexts immediately or work its way up to them.

The answer was less binary than I expected. Curriculum training consistently finished with lower validation loss across three seeds, especially at shorter contexts, but by the time both models had completed full-context training their 128-token performance was very close.

The part I found most useful was not that curriculum “won.” It was that my first pilot appeared to show a much stronger effect than the corrected experiment did. The first result was more exciting. The second result was more believable.

Learning to prefer the second one was probably the most useful part of Task B.
