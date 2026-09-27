import time
import statistics
import torch

from model import TinyLanguageModel
from model_kv_prealloc import KVCachePreallocLanguageModel
from tokenizers import Tokenizer


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/final_5pct.pt"
PROMPT = "Once upon a time"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
SEED = 42


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

eos_id = tokenizer.token_to_id("[EOS]")

base_model = TinyLanguageModel().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

base_model.load_state_dict(
    checkpoint["model_state_dict"]
)

base_model.eval()

kv_model = KVCachePreallocLanguageModel(
    base_model
).to(device)

kv_model.eval()


@torch.inference_mode()
def generate_once():
    torch.manual_seed(SEED)

    prompt_ids = tokenizer.encode(
        PROMPT
    ).ids

    x = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    logits = kv_model.prefill(x)

    generated_tokens = 0

    start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        next_token_logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        values, indices = torch.topk(
            next_token_logits,
            k=TOP_K,
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

        generated_tokens += 1

        if next_token.item() == eos_id:
            break

        logits = kv_model.forward_token(
            next_token
        )

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        elapsed,
        generated_tokens / elapsed,
        generated_tokens,
    )


print(
    "Optimization: "
    "KV cache with preallocated buffers"
)

print("\nWarming up...")

for i in range(WARMUP_RUNS):
    elapsed, tps, count = generate_once()

    print(
        f"Warm-up {i + 1}: "
        f"{elapsed:.4f}s, "
        f"{tps:.2f} tok/s"
    )


print("\nBenchmarking...")

latencies = []
throughputs = []

for i in range(BENCHMARK_RUNS):
    elapsed, tps, count = generate_once()

    latencies.append(elapsed)
    throughputs.append(tps)

    print(
        f"Run {i + 1}: "
        f"{elapsed:.4f}s, "
        f"{tps:.2f} tok/s, "
        f"{count} tokens"
    )


median_latency = statistics.median(
    latencies
)

median_tps = statistics.median(
    throughputs
)

mean_latency = statistics.mean(
    latencies
)

mean_tps = statistics.mean(
    throughputs
)


baseline_tps = 333.26

speedup = (
    median_tps / baseline_tps
)

improvement = (
    speedup - 1
) * 100


print(
    "\n=== KV PREALLOC RESULTS ==="
)

print(
    f"Median latency: "
    f"{median_latency:.4f} seconds"
)

print(
    f"Median tokens/sec: "
    f"{median_tps:.2f}"
)

print(
    f"Mean latency: "
    f"{mean_latency:.4f} seconds"
)

print(
    f"Mean tokens/sec: "
    f"{mean_tps:.2f}"
)

print(
    f"\nSpeedup vs baseline: "
    f"{speedup:.3f}x"
)

print(
    f"Throughput improvement: "
    f"{improvement:.2f}%"
)
