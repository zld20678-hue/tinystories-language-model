# TinyStories Language Model from Scratch

A small autoregressive Transformer language model trained from scratch on TinyStories as part of a technical learning exercise.

## Baseline model

- Parameters: ~5.29M
- Architecture: 4-layer Transformer
- Hidden dimension: 256
- Attention heads: 4
- Context length: 128 tokens
- Tokenizer: custom BPE tokenizer
- Vocabulary size: 4,096
- Training data: 21,197 TinyStories examples
- Validation data: 2,000 examples
- Hardware: Apple M4 using PyTorch MPS
- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 32

## Baseline training result

Validation loss decreased from approximately 8.3 to 3.55 after 1,000 optimization steps.

The model learned basic English sentence structure and TinyStories-style narrative patterns, but generated stories still showed weak long-range logical consistency.

## Baseline inference

Prompt:

`Once upon a time`

Baseline generation speed:

- 100 generated tokens
- 5.448 seconds
- 18.36 tokens/second

## Project files

- `train_tokenizer.py` — trains the custom BPE tokenizer
- `prepare_data.py` — converts text datasets into token IDs
- `model.py` — defines the Transformer language model
- `train.py` — training and validation loop
- `resume_train.py` — resumes training from a checkpoint
- `generate.py` — autoregressive text generation and inference benchmark

## Next steps

1. Continue training and evaluate convergence.
2. Compare generation quality at later checkpoints.
3. Profile inference bottlenecks.
4. Implement and benchmark inference optimization.
