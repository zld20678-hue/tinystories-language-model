from datasets import load_from_disk
from tokenizers import Tokenizer
import numpy as np

# 1. 读取我们已经训练好的 tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 2. 找到 [EOS] token 的数字 ID
eos_id = tokenizer.token_to_id("[EOS]")

# 3. 定义一个函数：
#    把一整个 dataset 的英文故事转换成 Token IDs
def encode_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)

    all_token_ids = []

    for text in dataset["text"]:
        ids = tokenizer.encode(text).ids

        all_token_ids.extend(ids)

        # 每篇故事结束后加入 [EOS]
        all_token_ids.append(eos_id)

    return np.array(all_token_ids, dtype=np.uint16)


# 4. 把训练教材全部转换成数字
print("Encoding training data...")
train_tokens = encode_dataset("data/tinystories_1pct")

# 5. 把 validation 考试卷也转换成数字
print("Encoding validation data...")
val_tokens = encode_dataset("data/tinystories_validation_2000")

# 6. 保存到硬盘
np.save("data/train_tokens.npy", train_tokens)
np.save("data/val_tokens.npy", val_tokens)

# 7. 打印结果
print("Done.")
print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
print("First 30 training token IDs:")
print(train_tokens[:30])
