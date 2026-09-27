from datasets import load_from_disk
from tokenizers import Tokenizer
import numpy as np


tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
eos_id = tokenizer.token_to_id("[EOS]")


def encode_dataset(dataset_path):
    """Flatten a saved text dataset into token IDs, separated by [EOS]."""
    dataset = load_from_disk(dataset_path)
    all_token_ids = []

    for text in dataset["text"]:
        ids = tokenizer.encode(text).ids
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    return np.array(all_token_ids, dtype=np.uint16)


print("Encoding training data...")
train_tokens = encode_dataset("data/tinystories_1pct")

print("Encoding validation data...")
val_tokens = encode_dataset("data/tinystories_validation_2000")

np.save("data/train_tokens.npy", train_tokens)
np.save("data/val_tokens.npy", val_tokens)

print("Done.")
print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
print("First 30 training token IDs:")
print(train_tokens[:30])
