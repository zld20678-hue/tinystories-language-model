import time
import statistics
import torch

from model import TinyLanguageModel
from tokenizers import Tokenizer


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/final_5pct.pt"
PROMPT = "Once upon a time"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5
SEED = 42


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

model = TinyLanguageModel().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


@torch.inference_mode()
def generate_batch(batch_size):
    torch.manual_seed(SEED)

    prompt_ids = tokenizer.encode(PROMPT).ids

    prompt_tensor = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    )

    x = prompt_tensor.unsqueeze(0).repeat(
        batch_size,
        1,
    )

    start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        x_context = x[:, -128:]

        logits = model(x_context)

        next_token_logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        values, indices = torch.topk(
            next_token_logits,
            k=TOP_K,
            dim=-1,
        )

        probabilities = torch.softmax(
            values,
            dim=-1,
        )

        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token = indices.gather(
            -1,
            sampled_index,
        )

        x = torch.cat(
            [x, next_token],
            dim=1,
        )

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    total_generated_tokens = (
        batch_size * MAX_NEW_TOKENS
    )

    total_tps = (
        total_generated_tokens / elapsed
    )

    per_sequence_tps = (
        MAX_NEW_TOKENS / elapsed
    )

    return (
        elapsed,
        total_tps,
        per_sequence_tps,
    )


print("Device:", device)
print("Checkpoint:", CHECKPOINT)

print("\n=== BATCHING BENCHMARK ===")

for batch_size in BATCH_SIZES:

    print(
        f"\n--- Batch size {batch_size} ---"
    )

    for i in range(WARMUP_RUNS):
        elapsed, total_tps, seq_tps = generate_batch(
            batch_size
        )

        print(
            f"Warm-up {i + 1}: "
            f"{elapsed:.4f}s, "
            f"{total_tps:.2f} total tok/s"
        )

    latencies = []
    total_throughputs = []
    per_sequence_throughputs = []

    for i in range(BENCHMARK_RUNS):
        elapsed, total_tps, seq_tps = generate_batch(
            batch_size
        )

        latencies.append(elapsed)
        total_throughputs.append(total_tps)
        per_sequence_throughputs.append(seq_tps)

        print(
            f"Run {i + 1}: "
            f"{elapsed:.4f}s, "
            f"{total_tps:.2f} total tok/s, "
            f"{seq_tps:.2f} tok/s per sequence"
        )

    median_latency = statistics.median(
        latencies
    )

    median_total_tps = statistics.median(
        total_throughputs
    )

    median_seq_tps = statistics.median(
        per_sequence_throughputs
    )

    print(
        f"Result batch={batch_size}: "
        f"median latency={median_latency:.4f}s, "
        f"total throughput={median_total_tps:.2f} tok/s, "
        f"per-sequence throughput={median_seq_tps:.2f} tok/s"
    )
