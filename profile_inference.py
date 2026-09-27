import time
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
WARMUP_RUNS = 3
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

eos_id = tokenizer.token_to_id("[EOS]")


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run_profile(measure=False):
    torch.manual_seed(SEED)

    token_ids = tokenizer.encode(PROMPT).ids

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    forward_time = 0.0
    sampling_time = 0.0
    concat_time = 0.0

    total_start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        x_context = x[:, -128:]

        sync()
        t0 = time.perf_counter()

        logits = model(x_context)

        sync()
        t1 = time.perf_counter()

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / TEMPERATURE

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
        t2 = time.perf_counter()

        x = torch.cat(
            [x, next_token],
            dim=1,
        )

        sync()
        t3 = time.perf_counter()

        if measure:
            forward_time += t1 - t0
            sampling_time += t2 - t1
            concat_time += t3 - t2

        if next_token.item() == eos_id:
            break

    sync()
    total_time = time.perf_counter() - total_start

    if not measure:
        return None

    measured = (
        forward_time
        + sampling_time
        + concat_time
    )

    other_time = total_time - measured

    return {
        "total": total_time,
        "forward": forward_time,
        "sampling": sampling_time,
        "concat": concat_time,
        "other": other_time,
    }


print("Device:", device)
print("Checkpoint:", CHECKPOINT)

print("\nWarming up...")

for i in range(WARMUP_RUNS):
    run_profile(measure=False)
    print(f"Warm-up {i + 1} complete")


print("\nProfiling...")

result = run_profile(measure=True)

total = result["total"]

print("\n=== PROFILE RESULTS ===")

for name in [
    "forward",
    "sampling",
    "concat",
    "other",
]:
    value = result[name]

    percentage = (
        value / total * 100
        if total > 0
        else 0
    )

    print(
        f"{name:>10}: "
        f"{value:.4f}s "
        f"({percentage:.1f}%)"
    )

print(
    f"\nTotal: {total:.4f}s"
)

print(
    f"Tokens/sec: "
    f"{MAX_NEW_TOKENS / total:.2f}"
)
