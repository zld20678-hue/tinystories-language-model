import torch

from model import TinyLanguageModel
from model_kv_prealloc import KVCachePreallocLanguageModel
from tokenizers import Tokenizer


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CHECKPOINT = "checkpoints/final_5pct.pt"
PROMPT = "Once upon a time"


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

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

prompt_ids = tokenizer.encode(PROMPT).ids

x = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device,
)

with torch.inference_mode():
    baseline_logits = base_model(x)
    baseline_last = baseline_logits[:, -1, :]

    kv_logits = kv_model.prefill(x)
    kv_last = kv_logits[:, -1, :]

difference = torch.abs(
    baseline_last - kv_last
)

print("=== TEST 1: PROMPT ===")
print(
    "Max absolute difference:",
    difference.max().item(),
)
print(
    "Mean absolute difference:",
    difference.mean().item(),
)

baseline_next = torch.argmax(
    baseline_last,
    dim=-1,
)

kv_next = torch.argmax(
    kv_last,
    dim=-1,
)

print(
    "Baseline next token:",
    tokenizer.decode(
        baseline_next.tolist()
    ),
)

print(
    "KV next token:",
    tokenizer.decode(
        kv_next.tolist()
    ),
)

next_token = baseline_next.view(1, 1)

extended_x = torch.cat(
    [x, next_token],
    dim=1,
)

with torch.inference_mode():
    baseline_extended = base_model(
        extended_x
    )

    baseline_extended_last = (
        baseline_extended[:, -1, :]
    )

    kv_extended = kv_model.forward_token(
        next_token
    )

    kv_extended_last = (
        kv_extended[:, -1, :]
    )

difference2 = torch.abs(
    baseline_extended_last
    - kv_extended_last
)

print("\n=== TEST 2: ONE DECODE STEP ===")
print(
    "Max absolute difference:",
    difference2.max().item(),
)
print(
    "Mean absolute difference:",
    difference2.mean().item(),
)

baseline_next2 = torch.argmax(
    baseline_extended_last,
    dim=-1,
)

kv_next2 = torch.argmax(
    kv_extended_last,
    dim=-1,
)

print(
    "Baseline next token:",
    tokenizer.decode(
        baseline_next2.tolist()
    ),
)

print(
    "KV next token:",
    tokenizer.decode(
        kv_next2.tolist()
    ),
)
