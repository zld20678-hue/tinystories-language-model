import os
import numpy as np
import torch
import torch.nn.functional as F

from model import TinyLanguageModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

train_data = np.load("data/train_tokens_5pct.npy")
val_data = np.load("data/val_tokens.npy")

batch_size = 32
seq_len = 128
learning_rate = 3e-4
max_steps = 6000


def get_batch(data):
    max_start = len(data) - seq_len - 1

    starts = np.random.randint(
        0,
        max_start,
        size=batch_size,
    )

    x = np.stack([
        data[i:i + seq_len]
        for i in starts
    ])

    y = np.stack([
        data[i + 1:i + seq_len + 1]
        for i in starts
    ])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y


@torch.no_grad()
def estimate_loss(data, eval_batches=20):
    model.eval()

    losses = []

    for _ in range(eval_batches):
        x, y = get_batch(data)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


model = TinyLanguageModel().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

os.makedirs("checkpoints", exist_ok=True)

print("Training from random initialization.")
print("Training tokens:", len(train_data))
print("Max steps:", max_steps)


for step in range(max_steps):
    model.train()

    x, y = get_batch(train_data)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 250 == 0:
        val_loss = estimate_loss(val_data)

        print(
            f"Step {step}: "
            f"train loss = {loss.item():.4f}, "
            f"val loss = {val_loss:.4f}"
        )

    if step > 0 and step % 1000 == 0:
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }

        torch.save(
            checkpoint,
            f"checkpoints/5pct_step_{step}.pt",
        )


final_val_loss = estimate_loss(val_data)

final_checkpoint = {
    "step": max_steps,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
    "val_loss": final_val_loss,
}

torch.save(
    final_checkpoint,
    "checkpoints/final_5pct.pt",
)

print(
    f"Final 5% checkpoint saved. "
    f"Validation loss = {final_val_loss:.4f}"
)
