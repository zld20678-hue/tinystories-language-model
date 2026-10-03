import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import TinyLanguageModel


BATCH_SIZE = 32
LEARNING_RATE = 3e-4
SEED = 42
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train_tokens_5pct.npy"
VAL_DATA_PATH = REPO_ROOT / "data" / "val_tokens.npy"
RESULTS_DIR = REPO_ROOT / "task_b" / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "task_b"

# Full-budget design: same total training tokens in both conditions.
FULL_SCHEDULES = {
    "baseline": [(128, 6000)],
    "curriculum": [(32, 8000), (64, 4000), (128, 2000)],
}

# 10% pilot with the same relative token allocation.
PILOT_SCHEDULES = {
    "baseline": [(128, 600)],
    "curriculum": [(32, 800), (64, 400), (128, 200)],
}

# Evaluate every 1,024,000 tokens in the full run; scaled to 102,400 in pilot.
EVAL_TOKEN_INTERVAL = {
    "full": 1_024_000,
    "pilot": 102_400,
}

EVAL_BATCHES = 20
VAL_CONTEXTS = (32, 64, 128)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_batch(data, seq_len, device, rng):
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=BATCH_SIZE)

    x = np.stack([data[i:i + seq_len] for i in starts])
    y = np.stack([data[i + 1:i + seq_len + 1] for i in starts])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(model, data, seq_len, device, seed, eval_batches=EVAL_BATCHES):
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(eval_batches):
        x, y = sample_batch(data, seq_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


def train_one_step(model, optimizer, train_data, seq_len, device, rng):
    x, y = sample_batch(train_data, seq_len, device, rng)
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint(path, model, optimizer, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def append_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(experiment, budget):
    device = choose_device()
    schedule_map = FULL_SCHEDULES if budget == "full" else PILOT_SCHEDULES
    schedule = schedule_map[experiment]
    eval_token_interval = EVAL_TOKEN_INTERVAL[budget]

    set_seed(SEED)
    train_rng = np.random.default_rng(SEED)

    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)

    model = TinyLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    run_name = f"{experiment}_{budget}_seed{SEED}"
    csv_path = RESULTS_DIR / f"{run_name}.csv"
    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path} already exists. Rename/delete it before rerunning so results are not mixed."
        )

    print("Task B context-length experiment")
    print("Experiment:", experiment)
    print("Budget:", budget)
    print("Device:", device)
    print("Seed:", SEED)
    print("Schedule:", schedule)
    print("Eval token interval:", f"{eval_token_interval:,}")

    total_steps = 0
    tokens_seen = 0
    next_eval_at = eval_token_interval
    start_time = time.perf_counter()

    # Initial evaluation before any optimizer updates.
    initial_vals = {
        seq_len: evaluate(model, val_data, seq_len, device, seed=10_000 + seq_len)
        for seq_len in VAL_CONTEXTS
    }
    append_row(
        csv_path,
        {
            "experiment": experiment,
            "budget": budget,
            "seed": SEED,
            "stage": 0,
            "global_step": 0,
            "stage_step": 0,
            "train_seq_len": 0,
            "tokens_seen": 0,
            "train_loss": "",
            "val_loss_32": initial_vals[32],
            "val_loss_64": initial_vals[64],
            "val_loss_128": initial_vals[128],
            "elapsed_seconds": 0.0,
            "learning_rate": LEARNING_RATE,
        },
    )

    last_train_loss = None

    for stage_index, (seq_len, stage_steps) in enumerate(schedule, start=1):
        print(f"\nStage {stage_index}: seq_len={seq_len}, steps={stage_steps}")
        tokens_per_step = BATCH_SIZE * seq_len

        for stage_step in range(1, stage_steps + 1):
            model.train()
            last_train_loss = train_one_step(
                model, optimizer, train_data, seq_len, device, train_rng
            )
            total_steps += 1
            tokens_seen += tokens_per_step

            if tokens_seen >= next_eval_at:
                synchronize(device)
                elapsed = time.perf_counter() - start_time

                vals = {
                    val_seq_len: evaluate(
                        model,
                        val_data,
                        val_seq_len,
                        device,
                        seed=10_000 + val_seq_len,
                    )
                    for val_seq_len in VAL_CONTEXTS
                }

                row = {
                    "experiment": experiment,
                    "budget": budget,
                    "seed": SEED,
                    "stage": stage_index,
                    "global_step": total_steps,
                    "stage_step": stage_step,
                    "train_seq_len": seq_len,
                    "tokens_seen": tokens_seen,
                    "train_loss": last_train_loss,
                    "val_loss_32": vals[32],
                    "val_loss_64": vals[64],
                    "val_loss_128": vals[128],
                    "elapsed_seconds": elapsed,
                    "learning_rate": LEARNING_RATE,
                }
                append_row(csv_path, row)

                print(
                    f"tokens={tokens_seen:,} | step={total_steps} | seq={seq_len} | "
                    f"train={last_train_loss:.4f} | "
                    f"val32={vals[32]:.4f} | val64={vals[64]:.4f} | val128={vals[128]:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )

                while next_eval_at <= tokens_seen:
                    next_eval_at += eval_token_interval

        checkpoint_path = CHECKPOINT_DIR / f"{run_name}_stage{stage_index}_seq{seq_len}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            {
                "experiment": experiment,
                "budget": budget,
                "seed": SEED,
                "stage": stage_index,
                "global_step": total_steps,
                "stage_step": stage_steps,
                "seq_len": seq_len,
                "tokens_seen": tokens_seen,
                "last_train_loss": last_train_loss,
            },
        )
        print("Saved stage checkpoint:", checkpoint_path)

    synchronize(device)
    total_elapsed = time.perf_counter() - start_time

    final_vals = {
        seq_len: evaluate(model, val_data, seq_len, device, seed=10_000 + seq_len)
        for seq_len in VAL_CONTEXTS
    }

    append_row(
        csv_path,
        {
            "experiment": experiment,
            "budget": budget,
            "seed": SEED,
            "stage": len(schedule),
            "global_step": total_steps,
            "stage_step": schedule[-1][1],
            "train_seq_len": schedule[-1][0],
            "tokens_seen": tokens_seen,
            "train_loss": last_train_loss,
            "val_loss_32": final_vals[32],
            "val_loss_64": final_vals[64],
            "val_loss_128": final_vals[128],
            "elapsed_seconds": total_elapsed,
            "learning_rate": LEARNING_RATE,
        },
    )

    final_checkpoint = CHECKPOINT_DIR / f"{run_name}_final.pt"
    save_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        {
            "experiment": experiment,
            "budget": budget,
            "seed": SEED,
            "global_step": total_steps,
            "tokens_seen": tokens_seen,
            "last_train_loss": last_train_loss,
            "val_loss_32": final_vals[32],
            "val_loss_64": final_vals[64],
            "val_loss_128": final_vals[128],
            "elapsed_seconds": total_elapsed,
        },
    )

    print("\n=== COMPLETE ===")
    print("Results:", csv_path)
    print("Final checkpoint:", final_checkpoint)
    print("Total steps:", total_steps)
    print("Tokens seen:", f"{tokens_seen:,}")
    print("Elapsed:", f"{total_elapsed / 60:.2f} min")
    print(
        "Final validation losses: "
        f"32={final_vals[32]:.4f}, 64={final_vals[64]:.4f}, 128={final_vals[128]:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("baseline", "curriculum"), required=True)
    parser.add_argument("--budget", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    run(args.experiment, args.budget)


if __name__ == "__main__":
    main()
