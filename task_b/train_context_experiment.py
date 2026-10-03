import argparse
import csv
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


MICROBATCH_SIZE = 32
EFFECTIVE_TOKENS_PER_UPDATE = 4096
LEARNING_RATE = 3e-4
SEED = 42
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train_tokens_5pct.npy"
VAL_DATA_PATH = REPO_ROOT / "data" / "val_tokens.npy"
RESULTS_DIR = REPO_ROOT / "task_b" / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "task_b"

# Each tuple is: (sequence length, optimizer updates, gradient-accumulation microbatches).
# Every optimizer update consumes the same effective token count (4096), so baseline
# and curriculum are matched on BOTH total tokens and number of parameter updates.
FULL_SCHEDULES = {
    "baseline": [(128, 6000, 1)],
    "curriculum": [(32, 2000, 4), (64, 2000, 2), (128, 2000, 1)],
}

# 10% pilot: 600 optimizer updates and 2,457,600 tokens in both conditions.
PILOT_SCHEDULES = {
    "baseline": [(128, 600, 1)],
    "curriculum": [(32, 200, 4), (64, 200, 2), (128, 200, 1)],
}

# Since every optimizer update represents 4,096 training tokens, these correspond
# to every 250 updates in the full run and every 25 updates in the pilot.
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
    starts = rng.integers(0, max_start, size=MICROBATCH_SIZE)

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


def train_one_update(model, optimizer, train_data, seq_len, accumulation_steps, device, rng):
    """One optimizer update with a fixed effective token budget of 4,096 tokens."""
    optimizer.zero_grad(set_to_none=True)
    losses = []

    for _ in range(accumulation_steps):
        x, y = sample_batch(train_data, seq_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        (loss / accumulation_steps).backward()
        losses.append(loss.item())

    optimizer.step()
    return float(np.mean(losses))


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

    # v2 marks the corrected design that controls both tokens and optimizer updates.
    run_name = f"{experiment}_{budget}_v2_seed{SEED}"
    csv_path = RESULTS_DIR / f"{run_name}.csv"
    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path} already exists. Rename/delete it before rerunning so results are not mixed."
        )

    print("Task B context-length experiment (controlled v2)")
    print("Experiment:", experiment)
    print("Budget:", budget)
    print("Device:", device)
    print("Seed:", SEED)
    print("Microbatch size:", MICROBATCH_SIZE)
    print("Effective tokens / optimizer update:", EFFECTIVE_TOKENS_PER_UPDATE)
    print("Schedule (seq_len, updates, accumulation):", schedule)
    print("Eval token interval:", f"{eval_token_interval:,}")

    total_updates = 0
    tokens_seen = 0
    next_eval_at = eval_token_interval
    start_time = time.perf_counter()

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
            "global_update": 0,
            "stage_update": 0,
            "train_seq_len": 0,
            "accumulation_steps": 0,
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

    for stage_index, (seq_len, stage_updates, accumulation_steps) in enumerate(schedule, start=1):
        tokens_per_update = MICROBATCH_SIZE * seq_len * accumulation_steps
        if tokens_per_update != EFFECTIVE_TOKENS_PER_UPDATE:
            raise ValueError(
                f"Stage seq={seq_len} consumes {tokens_per_update} tokens/update; "
                f"expected {EFFECTIVE_TOKENS_PER_UPDATE}."
            )

        print(
            f"\nStage {stage_index}: seq_len={seq_len}, updates={stage_updates}, "
            f"accumulation={accumulation_steps}"
        )

        for stage_update in range(1, stage_updates + 1):
            model.train()
            last_train_loss = train_one_update(
                model,
                optimizer,
                train_data,
                seq_len,
                accumulation_steps,
                device,
                train_rng,
            )
            total_updates += 1
            tokens_seen += EFFECTIVE_TOKENS_PER_UPDATE

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
                    "global_update": total_updates,
                    "stage_update": stage_update,
                    "train_seq_len": seq_len,
                    "accumulation_steps": accumulation_steps,
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
                    f"tokens={tokens_seen:,} | update={total_updates} | seq={seq_len} | "
                    f"accum={accumulation_steps} | train={last_train_loss:.4f} | "
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
                "global_update": total_updates,
                "stage_update": stage_updates,
                "seq_len": seq_len,
                "accumulation_steps": accumulation_steps,
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
            "global_update": total_updates,
            "stage_update": schedule[-1][1],
            "train_seq_len": schedule[-1][0],
            "accumulation_steps": schedule[-1][2],
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
            "global_update": total_updates,
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
    print("Total optimizer updates:", total_updates)
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
