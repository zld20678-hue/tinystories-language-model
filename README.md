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

## Milestone 2 — More training data

After observing diminishing returns on the 1% TinyStories subset, I kept the model architecture fixed and increased the training data to 5% of TinyStories.

### Controlled comparison

The following settings were kept unchanged:

- Model size: ~5.29M parameters
- Layers: 4
- Hidden dimension: 256
- Attention heads: 4
- Context length: 128
- Tokenizer vocabulary: 4,096
- Batch size: 32

Only the amount of training data was increased.

### Results

| Experiment | Training data | Steps | Validation loss |
| --- | ---: | ---: | ---: |
| Baseline | 1% TinyStories | 1,000 | ~3.55 |
| Extended training | 1% TinyStories | 4,000 | ~2.95 |
| More data | 5% TinyStories | 6,000 | 2.6264 |

The 1% model continued improving with more optimization steps, but the rate of improvement slowed substantially.

Increasing the dataset to 5% produced a larger improvement in validation performance than continuing to repeatedly train on the 1% subset.

The 5% model also generated more grammatically stable and locally coherent stories, although entity consistency and longer-range narrative logic remained imperfect.

Given the three-day scope of the task, I chose to stop scaling training here and move to inference optimization rather than continuing to brute-force model quality.
