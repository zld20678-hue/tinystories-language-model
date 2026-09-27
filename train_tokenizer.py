from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# 1. 读取我们的 21,197 篇训练故事
dataset = load_from_disk("data/tinystories_1pct")

# 2. 创建一个空白 tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 3. 告诉 tokenizer 如何先处理文字
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 4. 决定 tokenizer 最多学习多少种 token
trainer = BpeTrainer(
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# 5. 一批一批把训练故事交给 tokenizer
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]

# 6. 真正训练 tokenizer
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)
)

# 7. 设置如何把 token 重新还原成文字
tokenizer.decoder = ByteLevelDecoder()

# 8. 保存训练好的 tokenizer
tokenizer.save("tokenizer/tokenizer.json")

# 9. 测试一句话
test_text = "The puppy was playing in the garden."
encoded = tokenizer.encode(test_text)

print("Tokenizer training complete.")
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Original text:", test_text)
print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))
