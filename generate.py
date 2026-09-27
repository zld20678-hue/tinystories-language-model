import time
import torch

from model import TinyLanguageModel
from tokenizers import Tokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the tokenizer we trained earlier
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Recreate the same model architecture
model = TinyLanguageModel().to(device)

# Load the trained weights
checkpoint = torch.load(
    "checkpoints/final.pt",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded.")
print("Device:", device)


@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    # Convert human text into token IDs
    token_ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        # The model can only see at most 128 tokens
        x_context = x[:, -128:]

        # Forward pass
        logits = model(x_context)

        # We only care about the prediction after the last token
        next_token_logits = logits[:, -1, :]

        # Control randomness
        next_token_logits = next_token_logits / temperature

        # Keep only the top-k most likely choices
        values, indices = torch.topk(
            next_token_logits,
            k=top_k,
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

        # Add the predicted token back to the sequence
        x = torch.cat(
            [x, next_token],
            dim=1,
        )

        # Stop if the model predicts EOS
        if next_token.item() == tokenizer.token_to_id("[EOS]"):
            break

    elapsed = time.perf_counter() - start_time
    generated_ids = x[0].tolist()

    text = tokenizer.decode(generated_ids)

    generated_count = len(generated_ids) - len(token_ids)

    print("\n--- Generation ---")
    print(text)

    print("\n--- Performance ---")
    print("Generated tokens:", generated_count)
    print("Time:", round(elapsed, 3), "seconds")

    if elapsed > 0:
        print(
            "Tokens/sec:",
            round(generated_count / elapsed, 2),
        )


generate(
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
)
