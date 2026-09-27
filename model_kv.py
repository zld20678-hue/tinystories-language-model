import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCacheLanguageModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.token_embedding = base_model.token_embedding
        self.position_embedding = base_model.position_embedding
        self.layers = base_model.transformer.layers
        self.ln_f = base_model.ln_f
        self.lm_head = base_model.lm_head

        self.d_model = self.token_embedding.embedding_dim

        first_layer = self.layers[0]

        self.n_heads = first_layer.self_attn.num_heads
        self.head_dim = self.d_model // self.n_heads

        self.max_seq_len = (
            self.position_embedding.num_embeddings
        )

        self.reset_cache()


    def reset_cache(self):
        self.k_cache = [None for _ in self.layers]
        self.v_cache = [None for _ in self.layers]
        self.position = 0


    def _attention_step(self, layer, x, layer_idx):
        """
        x shape:
        [batch, 1, d_model]

        Computes attention only for the newest token,
        while reusing cached K/V from previous tokens.
        """

        residual = x

        # norm_first=True in the original model
        x_norm = layer.norm1(x)

        attn = layer.self_attn

        # Original PyTorch MultiheadAttention stores
        # Q, K, V projections together.
        qkv = F.linear(
            x_norm,
            attn.in_proj_weight,
            attn.in_proj_bias,
        )

        q, k, v = qkv.chunk(3, dim=-1)

        batch_size = x.shape[0]

        q = q.view(
            batch_size,
            1,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            1,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            1,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Add newest K/V to the cache.
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], k],
                dim=2,
            )

            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], v],
                dim=2,
            )

        cached_k = self.k_cache[layer_idx]
        cached_v = self.v_cache[layer_idx]

        # Q only comes from the newest token.
        # K/V contain every token seen so far.
        scores = torch.matmul(
            q,
            cached_k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.matmul(
            probabilities,
            cached_v,
        )

        context = context.transpose(
            1,
            2,
        ).contiguous()

        context = context.view(
            batch_size,
            1,
            self.d_model,
        )

        attention_output = F.linear(
            context,
            attn.out_proj.weight,
            attn.out_proj.bias,
        )

        # First residual connection
        x = residual + layer.dropout1(
            attention_output
        )

        # Feed-forward block
        residual = x

        ff = layer.norm2(x)

        ff = layer.linear1(ff)
        ff = layer.activation(ff)
        ff = layer.dropout(ff)
        ff = layer.linear2(ff)
        ff = layer.dropout2(ff)

        x = residual + ff

        return x


    @torch.inference_mode()
    def forward_token(self, token_id):
        """
        Process exactly ONE new token.

        token_id shape:
        [batch, 1]
        """

        if self.position >= self.max_seq_len:
            raise RuntimeError(
                "KV cache reached max_seq_len."
            )

        position_tensor = torch.tensor(
            [self.position],
            device=token_id.device,
            dtype=torch.long,
        )

        token_emb = self.token_embedding(
            token_id
        )

        position_emb = self.position_embedding(
            position_tensor
        )

        x = token_emb + position_emb

        for layer_idx, layer in enumerate(
            self.layers
        ):
            x = self._attention_step(
                layer,
                x,
                layer_idx,
            )

        x = self.ln_f(x)

        logits = self.lm_head(x)

        self.position += 1

        return logits


    @torch.inference_mode()
    def prefill(self, token_ids):
        """
        Fill the KV cache with the prompt.

        token_ids shape:
        [batch, seq_len]
        """

        self.reset_cache()

        logits = None

        for i in range(token_ids.shape[1]):
            logits = self.forward_token(
                token_ids[:, i:i+1]
            )

        return logits
