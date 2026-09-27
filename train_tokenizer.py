from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# Load the 21,197-story training subset used for the first experiments.
dataset = load_from_disk("data/tinystories_1pct")

# Start with an empty BPE tokenizer and learn the vocabulary from TinyStories.
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
)


def batch_iterator(batch_size=1000):
    """Yield text in chunks so tokenizer training does not load it all at once."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]


tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset),
)

tokenizer.decoder = ByteLevelDecoder()
tokenizer.save("tokenizer/tokenizer.json")

# Small round-trip check: text -> tokens -> IDs -> decoded text.
test_text = "The puppy was playing in the garden."
encoded = tokenizer.encode(test_text)

print("Tokenizer training complete.")
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Original text:", test_text)
print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))
