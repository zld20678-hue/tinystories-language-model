# Learning Log

## Milestone 1 — Baseline language model

### Dataset
I used TinyStories because it provides simple, structured English narratives that are suitable for training and inspecting a small language model.

Training subset:
- 21,197 stories
- approximately 4.83M preprocessed tokens

Validation subset:
- 2,000 stories
- approximately 398K preprocessed tokens

### Tokenizer
I trained a custom BPE tokenizer with a vocabulary size of 4,096.

The tokenizer converts raw text into tokens and then maps each token to an integer token ID.

### Model
The baseline Transformer uses:
- 4 Transformer layers
- 256-dimensional token representations
- 4 attention heads
- 128-token context length
- approximately 5.29M trainable parameters

### Training
Optimizer:
- AdamW

Learning rate:
- 3e-4

Batch:
- 32 sequences × 128 tokens

A 20-step sanity test reduced loss from about 8.50 to 6.47, confirming that forward propagation, loss computation, backpropagation, and optimization were working.

The full 1,000-step run reduced validation loss from about 8.30 to about 3.55.

Training and validation loss remained close, with no clear evidence of severe overfitting.

### Generation
The first trained model produced locally coherent English and learned common TinyStories patterns such as character introductions, dialogue, and simple narrative transitions.

However, entity consistency and longer-range logic remained weak.

### Baseline inference benchmark
Hardware:
- Apple M4
- PyTorch MPS

Generation:
- 100 tokens
- 5.448 seconds
- 18.36 tokens/sec

This result will serve as the baseline for inference optimization.

## Milestone 2 — Increasing training data

After extending the original 1% dataset model from 1,000 to 4,000 steps, validation loss improved from about 3.55 to about 2.95.

However, the rate of improvement slowed significantly, suggesting diminishing returns from repeatedly training on the same small subset.

I therefore changed one experimental variable: dataset size.

The model architecture, tokenizer, context length, optimizer family, and hardware remained fixed, while the training subset increased from 1% to 5% of TinyStories.

The 5% dataset contained:

- 23,775,548 training tokens
- approximately 5,805 batch-equivalent steps at 32 × 128 tokens per step

I trained the model from random initialization for 6,000 steps.

Final validation loss:

- 5% model: 2.6264

The generated stories showed better grammar and local coherence than the smaller-data models, although entity consistency and long-range story logic were still limited.

At this point I stopped further model scaling and moved to inference optimization, since the main training objective had been demonstrated successfully.
