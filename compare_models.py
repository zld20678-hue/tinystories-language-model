import time
import torch

from model import TinyLanguageModel
from tokenizers import Tokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
SEED = 42


def load_model(checkpoint_path):
    model = TinyLanguageModel().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def generate(model, prompt):
    torch.manual_seed(SEED)

    token_ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    start_time = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        x_context = x[:, -128:]

        logits = model(x_context)

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / TEMPERATURE

        values, indices = torch.topk(
            next_token_logits,
            k=TOP_K,
        )

        probabilities = torch.softmax(values, dim=-1)

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

        if next_token.item() == tokenizer.token_to_id("[EOS]"):
            break

    elapsed = time.perf_counter() - start_time

    generated_ids = x[0].tolist()

    text = tokenizer.decode(generated_ids)

    generated_count = len(generated_ids) - len(token_ids)

    return text, generated_count, elapsed


baseline_model = load_model("checkpoints/final.pt")
model_2000 = load_model("checkpoints/final_2000.pt")


print("\n==============================")
print("1000-STEP MODEL")
print("==============================")

text, count, elapsed = generate(
    baseline_model,
    PROMPT,
)

print(text)
print()
print("Generated tokens:", count)
print("Time:", round(elapsed, 3), "seconds")
print("Tokens/sec:", round(count / elapsed, 2))


print("\n==============================")
print("2000-STEP MODEL")
print("==============================")

text, count, elapsed = generate(
    model_2000,
    PROMPT,
)

print(text)
print()
print("Generated tokens:", count)
print("Time:", round(elapsed, 3), "seconds")
print("Tokens/sec:", round(count / elapsed, 2))
