from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "task_b" / "results"
FIGURES_DIR = REPO_ROOT / "task_b" / "figures"
SEEDS = [42, 7, 123]
CONTEXTS = [32, 64, 128]
TRANSITIONS = [8_192_000, 16_384_000]


def read_csv(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in [
                "seed",
                "stage",
                "global_update",
                "stage_update",
                "train_seq_len",
                "accumulation_steps",
                "tokens_seen",
            ]:
                parsed[key] = int(parsed[key])
            for key in [
                "val_loss_32",
                "val_loss_64",
                "val_loss_128",
                "elapsed_seconds",
                "learning_rate",
            ]:
                parsed[key] = float(parsed[key])
            parsed["train_loss"] = (
                float(parsed["train_loss"]) if parsed["train_loss"] else None
            )
            rows.append(parsed)
    return rows


def load_runs():
    runs = {}
    for seed in SEEDS:
        for experiment in ["baseline", "curriculum"]:
            path = RESULTS_DIR / f"{experiment}_full_v2_seed{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing expected result file: {path}")
            runs[(experiment, seed)] = read_csv(path)
    return runs


def dedupe_by_tokens(rows):
    # The runner writes a final row at the same token count as the last scheduled
    # evaluation. Keep the final occurrence so each x-value appears once.
    by_tokens = {}
    for row in rows:
        by_tokens[row["tokens_seen"]] = row
    return [by_tokens[k] for k in sorted(by_tokens)]


def mean_curve(runs, experiment, metric):
    cleaned = {
        seed: {r["tokens_seen"]: r for r in dedupe_by_tokens(runs[(experiment, seed)])}
        for seed in SEEDS
    }
    common_tokens = sorted(set.intersection(*[set(v.keys()) for v in cleaned.values()]))
    means = []
    for token_count in common_tokens:
        vals = [cleaned[seed][token_count][metric] for seed in SEEDS]
        means.append(statistics.mean(vals))
    return common_tokens, means


def final_row(rows):
    return dedupe_by_tokens(rows)[-1]


def write_summary(runs):
    summary_path = RESULTS_DIR / "context_curriculum_summary.csv"
    fieldnames = [
        "seed",
        "context",
        "baseline_final_loss",
        "curriculum_final_loss",
        "baseline_minus_curriculum",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in SEEDS:
            baseline = final_row(runs[("baseline", seed)])
            curriculum = final_row(runs[("curriculum", seed)])
            for context in CONTEXTS:
                key = f"val_loss_{context}"
                b = baseline[key]
                c = curriculum[key]
                writer.writerow(
                    {
                        "seed": seed,
                        "context": context,
                        "baseline_final_loss": f"{b:.6f}",
                        "curriculum_final_loss": f"{c:.6f}",
                        "baseline_minus_curriculum": f"{b-c:.6f}",
                    }
                )
    return summary_path


def plot_val128_dynamics(runs):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for experiment in ["baseline", "curriculum"]:
        for seed in SEEDS:
            rows = dedupe_by_tokens(runs[(experiment, seed)])
            ax.plot(
                [r["tokens_seen"] / 1e6 for r in rows],
                [r["val_loss_128"] for r in rows],
                alpha=0.22,
                linewidth=1.2,
            )

        tokens, means = mean_curve(runs, experiment, "val_loss_128")
        ax.plot(
            [t / 1e6 for t in tokens],
            means,
            linewidth=2.8,
            label=f"{experiment.capitalize()} mean",
        )

    for transition in TRANSITIONS:
        ax.axvline(transition / 1e6, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.text(TRANSITIONS[0] / 1e6, ax.get_ylim()[1], " 32→64", va="top")
    ax.text(TRANSITIONS[1] / 1e6, ax.get_ylim()[1], " 64→128", va="top")
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Validation loss @ 128-token context")
    ax.set_title("Task B: Full-context validation dynamics across 3 seeds")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "val128_vs_tokens.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_final_losses(runs):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_positions = {32: 0, 64: 1, 128: 2}
    offsets = {"baseline": -0.12, "curriculum": 0.12}

    for experiment in ["baseline", "curriculum"]:
        means = []
        stds = []
        xs = []
        for context in CONTEXTS:
            values = [
                final_row(runs[(experiment, seed)])[f"val_loss_{context}"]
                for seed in SEEDS
            ]
            x = x_positions[context] + offsets[experiment]
            xs.append(x)
            means.append(statistics.mean(values))
            stds.append(statistics.stdev(values))
            ax.scatter([x] * len(values), values, alpha=0.55, s=35)

        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2,
            label=experiment.capitalize(),
        )

    ax.set_xticks([0, 1, 2], ["32", "64", "128"])
    ax.set_xlabel("Validation context length")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Final validation loss across 3 seeds (mean ± SD)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "final_loss_by_context.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_curriculum_context_transfer(runs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for context in CONTEXTS:
        tokens, means = mean_curve(runs, "curriculum", f"val_loss_{context}")
        ax.plot(
            [t / 1e6 for t in tokens],
            means,
            linewidth=2.2,
            label=f"Validation @ {context}",
        )

    for transition in TRANSITIONS:
        ax.axvline(transition / 1e6, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(TRANSITIONS[0] / 1e6, ax.get_ylim()[1], " 32→64", va="top")
    ax.text(TRANSITIONS[1] / 1e6, ax.get_ylim()[1], " 64→128", va="top")
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Curriculum run: transfer across context lengths (3-seed mean)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "curriculum_context_transfer.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def print_final_statistics(runs):
    print("\n=== FINAL FULL-RUN SUMMARY (3 seeds) ===")
    for context in CONTEXTS:
        baseline = [
            final_row(runs[("baseline", seed)])[f"val_loss_{context}"] for seed in SEEDS
        ]
        curriculum = [
            final_row(runs[("curriculum", seed)])[f"val_loss_{context}"] for seed in SEEDS
        ]
        differences = [b - c for b, c in zip(baseline, curriculum)]
        print(
            f"context={context:>3} | "
            f"baseline={statistics.mean(baseline):.4f} ± {statistics.stdev(baseline):.4f} | "
            f"curriculum={statistics.mean(curriculum):.4f} ± {statistics.stdev(curriculum):.4f} | "
            f"mean improvement={statistics.mean(differences):.4f}"
        )


def main():
    runs = load_runs()
    print_final_statistics(runs)
    summary = write_summary(runs)
    fig1 = plot_val128_dynamics(runs)
    fig2 = plot_final_losses(runs)
    fig3 = plot_curriculum_context_transfer(runs)

    print("\nSaved:")
    print(summary)
    print(fig1)
    print(fig2)
    print(fig3)


if __name__ == "__main__":
    main()
