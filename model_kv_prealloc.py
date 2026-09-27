import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCachePreallocLanguageModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.token_embedding = base_model.token_embedding
        self.position_embedding = base_model.position_embedding
        self.layers = base_model.transformer.layers
        self.ln_f = base_model.ln_f
        self.lm_head = base_model.lm_head

        self.d_model = self.token_embedding.embedding_dim
        self.max_seq_len = self.position_embedding.num_embeddings

        first_layer = self.layers[0]
        self.n_heads = first_layer.self_attn.num_heads
        self.head_dim = self.d_model // self.n_heads

        self.k_cache = None
        self.v_cache = None
        self.position = 0


    def reset_cache(self, batch_size, device, dtype):
        self.k_cache = []
        self.v_cache = []

        for _ in self.layers:
            k = torch.empty(
                batch_size,
                self.n_heads,
                self.max_seq_len,
                self.head_dim,
                device=device,
                dtype=dtype,
            )

            v = torch.empty(
                batch_size,
                self.n_heads,
                self.max_seq_len,
                self.head_dim,
                device=device,
                dtype=dtype,
            )

            self.k_cache.append(k)
            self.v_cache.append(v)

        self.position = 0


    def _attention_step(self, layer, x, layer_idx):
        residual = x

        x_norm = layer.norm1(x)

        attn = layer.self_attn

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

        # Write newest K/V directly into preallocated cache.
        self.k_cache[layer_idx][
            :,
            :,
            self.position:self.position + 1,
            :
        ] = k

        self.v_cache[layer_idx][
            :,
            :,
            self.position:self.position + 1,
            :
        ] = v

        cached_k = self.k_cache[layer_idx][
            :,
            :,
            :self.position + 1,
            :
        ]

        cached_v = self.v_cache[layer_idx][
            :,
            :,
            :self.position + 1,
            :
        ]

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

        x = residual + layer.dropout1(
            attention_output
        )

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
        if self.position >= self.max_seq_len:
            raise RuntimeError(
                "KV cache reached max_seq_len."
            )

        if self.k_cache is None:
            self.reset_cache(
                batch_size=token_id.shape[0],
                device=token_id.device,
                dtype=self.token_embedding.weight.dtype,
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
        self.reset_cache(
            batch_size=token_ids.shape[0],
            device=token_ids.device,
            dtype=self.token_embedding.weight.dtype,
        )

        logits = None

        for i in range(token_ids.shape[1]):
            logits = self.forward_token(
                token_ids[:, i:i + 1]
            )

        return logits
