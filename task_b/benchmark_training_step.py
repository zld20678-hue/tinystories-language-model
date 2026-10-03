import time
import numpy as np
import torch
import torch.nn.functional as F

from model import TinyLanguageModel


BATCH_SIZE = 32
SEQ_LENGTHS = [32, 64, 128]
WARMUP_STEPS = 20
TIMED_STEPS = 200
SEED = 42
TRAIN_DATA_PATH = "data/train_tokens_5pct.npy"


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def get_batch(data, seq_len, device, rng):
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=BATCH_SIZE)

    x = np.stack([data[i:i + seq_len] for i in starts])
    y = np.stack([data[i + 1:i + seq_len + 1] for i in starts])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y


def train_step(model, optimizer, x, y):
    model.train()
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def benchmark_seq_len(data, seq_len, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = TinyLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"\n--- seq_len={seq_len} ---")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Timed steps:  {TIMED_STEPS}")

    for _ in range(WARMUP_STEPS):
        x, y = get_batch(data, seq_len, device, rng)
        train_step(model, optimizer, x, y)

    synchronize(device)
    start = time.perf_counter()

    last_loss = None
    for _ in range(TIMED_STEPS):
        x, y = get_batch(data, seq_len, device, rng)
        last_loss = train_step(model, optimizer, x, y)

    synchronize(device)
    elapsed = time.perf_counter() - start

    seconds_per_step = elapsed / TIMED_STEPS
    tokens_per_step = BATCH_SIZE * seq_len
    training_tokens_per_second = tokens_per_step / seconds_per_step

    result = {
        "seq_len": seq_len,
        "seconds_per_step": seconds_per_step,
        "training_tokens_per_second": training_tokens_per_second,
        "last_loss": last_loss,
    }

    print(f"Elapsed:                 {elapsed:.3f} s")
    print(f"Seconds / step:          {seconds_per_step:.6f}")
    print(f"Training tokens / sec:   {training_tokens_per_second:,.2f}")
    print(f"Last training loss:      {last_loss:.4f}")

    del model, optimizer, x, y
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Task B training-step timing benchmark")
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Data:", TRAIN_DATA_PATH)

    data = np.load(TRAIN_DATA_PATH)
    results = [benchmark_seq_len(data, seq_len, device) for seq_len in SEQ_LENGTHS]

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"seq={r['seq_len']:>3} | "
            f"sec/step={r['seconds_per_step']:.6f} | "
            f"train tok/s={r['training_tokens_per_second']:,.2f}"
        )

    t32 = results[0]["seconds_per_step"]
    t64 = results[1]["seconds_per_step"]
    t128 = results[2]["seconds_per_step"]

    baseline_seconds = 6000 * t128
    curriculum_seconds = 8000 * t32 + 4000 * t64 + 2000 * t128

    print("\n=== ESTIMATED CORE TRAINING TIME ===")
    print(
        "Baseline (128 x 6000): "
        f"{baseline_seconds / 3600:.2f} hours"
    )
    print(
        "Curriculum (32x8000 + 64x4000 + 128x2000): "
        f"{curriculum_seconds / 3600:.2f} hours"
    )
    print(
        "Combined core training time: "
        f"{(baseline_seconds + curriculum_seconds) / 3600:.2f} hours"
    )
    print("Note: validation/checkpoint overhead is not included yet.")


if __name__ == "__main__":
    main()
