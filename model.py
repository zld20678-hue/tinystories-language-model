import torch
import torch.nn as nn


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=4096,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=128,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.shape

        positions = torch.arange(seq_len, device=idx.device)

        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=idx.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits
