# Task B — Context-Length Curriculum

This folder contains the second half of the project: a controlled training-dynamics experiment built on the 5.29M-parameter TinyStories model from Task A.

## Question

> Does gradually increasing context length during training change learning efficiency or final validation performance compared with training at the full 128-token context from the start?

I compared:

- **Baseline:** 128-token context throughout training
- **Curriculum:** 32 → 64 → 128 tokens

The corrected full experiment controls both total training tokens and optimizer-update count:

- 24,576,000 training tokens
- 6,000 optimizer updates
- 4,096 effective tokens per update
- seeds 42, 7, and 123

Gradient accumulation keeps the effective token budget per update constant across context lengths.

## Main result

Across three seeds, curriculum training finished with lower mean validation loss at all three evaluation contexts:

| Validation context | Baseline mean | Curriculum mean | Mean improvement |
|---|---:|---:|---:|
| 32 | 2.8216 | 2.7825 | 0.0391 |
| 64 | 2.7354 | 2.7093 | 0.0262 |
| 128 | 2.6294 | 2.6227 | 0.0067 |

The more interesting result was the trajectory: curriculum fell behind on 128-token validation while training on short sequences, then caught up rapidly once 128-token training began. The strongest effect remained at shorter contexts; the final full-context advantage was small.

My main conclusion is therefore:

> **Context-length curriculum changed the path of learning much more than it changed the final long-context endpoint.**

## Files

- `train_context_experiment.py` — controlled baseline/curriculum runner
- `run_seed.py` — replication runner for additional random seeds
- `benchmark_training_step.py` — M4 timing benchmark used before the full experiment
- `analyze_context_results.py` — aggregates the three seeds and produces the summary/figures
- `REPORT.md` — final English technical report
- `REPORT_zh.md` — Chinese version of the report
- `results/` — full-run CSVs and summary table
- `figures/` — generated analysis figures

## Reproduce the full experiment

From the repository root:

```bash
python task_b/train_context_experiment.py --experiment baseline --budget full
python task_b/train_context_experiment.py --experiment curriculum --budget full

python task_b/run_seed.py --experiment baseline --budget full --seed 7
python task_b/run_seed.py --experiment curriculum --budget full --seed 7
python task_b/run_seed.py --experiment baseline --budget full --seed 123
python task_b/run_seed.py --experiment curriculum --budget full --seed 123

python task_b/analyze_context_results.py
```

The original pilot is not used in the final statistics. It exposed an experimental-design confound: matching total tokens alone gave the curriculum condition more optimizer updates. The final design fixes that with gradient accumulation and matches both tokens and update count.
