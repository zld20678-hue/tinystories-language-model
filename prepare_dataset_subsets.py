from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")
TRAIN_1PCT_SIZE = 21_197
TRAIN_5PCT_SIZE = 105_986
VALIDATION_SIZE = 2_000


DATA_DIR.mkdir(exist_ok=True)

print("Downloading TinyStories from Hugging Face...")
train = load_dataset("roneneldan/TinyStories", split="train")
validation = load_dataset("roneneldan/TinyStories", split="validation")

print("Creating local subsets expected by the training scripts...")
train.select(range(TRAIN_1PCT_SIZE)).save_to_disk(
    DATA_DIR / "tinystories_1pct"
)
train.select(range(TRAIN_5PCT_SIZE)).save_to_disk(
    DATA_DIR / "tinystories_5pct"
)
validation.select(range(VALIDATION_SIZE)).save_to_disk(
    DATA_DIR / "tinystories_validation_2000"
)

print("Done.")
print(f"1% training stories: {TRAIN_1PCT_SIZE:,}")
print(f"5% training stories: {TRAIN_5PCT_SIZE:,}")
print(f"Validation stories: {VALIDATION_SIZE:,}")
