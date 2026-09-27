import time
import torch

from model import TinyLanguageModel
from model_kv import KVCacheLanguageModel
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

kv_model = KVCacheLanguageModel(
    base_model
).to(device)

kv_model.eval()


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run_once(measure=False):
    torch.manual_seed(SEED)

    prompt_ids = tokenizer.encode(PROMPT).ids

    x = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    sync()
    total_start = time.perf_counter()

    # -------------------------
    # Prefill
    # -------------------------
    sync()
    t0 = time.perf_counter()

    logits = kv_model.prefill(x)

    sync()
    t1 = time.perf_counter()

    prefill_time = t1 - t0

    attention_time = 0.0
    sampling_time = 0.0
    cache_update_time = 0.0

    generated_tokens = 0

    for _ in range(MAX_NEW_TOKENS):

        # -------------------------
        # Sampling
        # -------------------------
        sync()
        s0 = time.perf_counter()

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

        sync()
        s1 = time.perf_counter()

        sampling_time += s1 - s0

        generated_tokens += 1

        if next_token.item() == eos_id:
            break

        # -------------------------
        # Incremental KV decode
        # -------------------------
        sync()
        k0 = time.perf_counter()

        logits = kv_model.forward_token(
            next_token
        )

        sync()
        k1 = time.perf_counter()

        attention_time += k1 - k0


    sync()
    total_time = time.perf_counter() - total_start

    if not measure:
        return None

    measured = (
        prefill_time
        + attention_time
        + sampling_time
    )

    other_time = total_time - measured

    return {
        "total": total_time,
        "prefill": prefill_time,
        "incremental_decode": attention_time,
        "sampling": sampling_time,
        "other": other_time,
        "tokens": generated_tokens,
    }


print("Device:", device)
print("Profiling KV-cache inference")

print("\nWarming up...")

for i in range(WARMUP_RUNS):
    run_once(measure=False)
    print(f"Warm-up {i + 1} complete")


print("\nProfiling...")

result = run_once(measure=True)

total = result["total"]

print("\n=== KV PROFILE RESULTS ===")

for name in [
    "prefill",
    "incremental_decode",
    "sampling",
    "other",
]:
    value = result[name]

    percentage = (
        value / total * 100
        if total > 0
        else 0
    )

    print(
        f"{name:>20}: "
        f"{value:.4f}s "
        f"({percentage:.1f}%)"
    )

print(
    f"\nTotal: {total:.4f}s"
)

print(
    f"Tokens/sec: "
    f"{result['tokens'] / total:.2f}"
)
