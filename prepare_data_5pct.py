from datasets import load_from_disk
from tokenizers import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

eos_id = tokenizer.token_to_id("[EOS]")


def encode_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)

    all_token_ids = []

    for text in dataset["text"]:
        ids = tokenizer.encode(text).ids

        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    return np.array(all_token_ids, dtype=np.uint16)


print("Encoding 5% training data...")

train_tokens = encode_dataset(
    "data/tinystories_5pct"
)

np.save(
    "data/train_tokens_5pct.npy",
    train_tokens,
)

print("Done.")
print("Training tokens:", len(train_tokens))
print("File saved to: data/train_tokens_5pct.npy")
