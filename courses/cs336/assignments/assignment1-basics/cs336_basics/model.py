import math

import torch
from torch import nn

def silu(x : torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    maximum = x.max(dim = dim, keepdim = True).values
    shifted = x - maximum
    exp_values = torch.exp(shifted)
    denominator = exp_values.sum(dim = dim, keepdim = True)

    return exp_values / denominator

def scaled_dot_product_attention(
    Q : torch.Tensor,
    K : torch.Tensor,
    V : torch.Tensor,
    mask : torch.Tensor | None = None,
):
    d_k = Q.shape[-1]

    score = Q @ K.transpose(-2, -1)
    scores = score / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weight = softmax(scores, dim = -1)

    return attention_weight @ V

class Linear(nn.Module):

    def __init__(
        self,
        d_in : int,
        d_out : int,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                d_out,
                d_in,
                device = device,
                dtype = dtype,
            )
        )

        std = math.sqrt(2 / (d_in + d_out))

        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        )

    def forward(self, x : torch.Tensor):
        return x @ self.weight.transpose(-2, -1)

class Embedding(nn.Module):

    def __init__(
        self,
        vocab_size : int,
        d_model : int,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                vocab_size,
                d_model,
                device = device,
                dtype = dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1.0,
            a = -3.0,
            b = 3.0,
        )

    def forward(self, token_ids : torch.Tensor):
        return self.weight[token_ids]

class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model : int,
        eps : float = 1e-5,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device = device, dtype = dtype)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        x_float = x.to(torch.float32)

        mean_square = x_float.square().mean(
            dim = -1,
            keepdim = True,
        )

        rms = torch.sqrt(mean_square + self.eps)

        output = (x_float / rms) * self.weight

        return output.to(original_dtype)

class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model : int,
        d_ff : int,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.w1 = Linear(
            d_model,
            d_ff,
            device = device,
            dtype = dtype,
        )
        self.w3 = Linear(
            d_model,
            d_ff,
            device = device,
            dtype = dtype,
        )
        self.w2 = Linear(
            d_ff,
            d_model,
            device = device,
            dtype = dtype,
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        content = self.w3(x)
        hidden = gate * content
        return self.w2(hidden)

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model : int,
        num_heads : int,
        device = None,
        dtype = None,
        rope = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.d_head = d_model // num_heads

        self.q_proj = Linear(
            d_model,
            d_model,
            device = device,
            dtype = dtype,
        )
        self.k_proj = Linear(
            d_model,
            d_model,
            device = device,
            dtype = dtype
        )
        self.v_proj = Linear(
            d_model,
            d_model,
            device = device,
            dtype = dtype
        )
        self.output_proj = Linear(
            d_model,
            d_model,
            device = device,
            dtype = dtype
        )

    def _split_heads(self, x : torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[: -1], self.num_heads, self.d_head)

        return x.transpose(-3, -2)

    def forward(self, x : torch.Tensor, token_positions : torch.Tensor | None = None) -> torch.Tensor:
        batch_shape = x.shape[: -2]
        seq_len = x.shape[-2]

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            device = x.device,
            dtype = torch.bool,
        ).tril()

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device = x.device)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        head_outputs = scaled_dot_product_attention(Q, K, V, mask = causal_mask)

        combined = head_outputs.transpose(-3, -2)
        combined = combined.reshape(*batch_shape, seq_len, self.d_model)

        return self.output_proj(combined)

class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta : int,
        d_k : int,
        max_seq_len : int,
        device = None,
    ):
        super().__init__()

        self.d_k = d_k

        pair_indices = torch.arange(0, d_k, 2, device = device, dtype = torch.float32)
        inv_freq = theta ** (-pair_indices / d_k)

        positions = torch.arange(max_seq_len, device = device, dtype = torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cache", torch.cos(angles), persistent = False
        )
        self.register_buffer(
            "sin_cache", torch.sin(angles), persistent = False
        )

    def forward(
        self,
        x : torch.Tensor,
        token_positions : torch.Tensor
    ) -> torch.Tensor:

        positions = token_positions.to(
            device = self.cos_cache.device,
            dtype = torch.long
        )
        cos = self.cos_cache[positions].to(
            device = x.device,
            dtype = x.dtype
        )
        sin = self.sin_cache[positions].to(
            device = x.device,
            dtype = x.dtype
        )

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        return torch.stack(
            (rotated_even, rotated_odd), dim = -1
        ).flatten(-2)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model : int,
        num_heads : int,
        d_ff : int,
        max_seq_len : int,
        theta : float,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)

        rope = RotaryPositionalEmbedding(
            theta = theta,
            d_k = d_model // num_heads,
            max_seq_len = max_seq_len,
            device = device
        )

        self.attn = MultiHeadSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            device = device,
            dtype = dtype,
            rope = rope,
        )

        self.ffn = SwiGLU(
            d_model = d_model,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
        )

    def forward(
        self,
        x : torch.Tensor,
        token_positions : torch.Tensor | None = None,
    ) -> torch.Tensor:

        attention_output = self.attn(
            self.ln1(x),
            token_positions = token_positions
        )
        h = x + attention_output

        ffn_output = self.ffn(self.ln2(h))
        y = h + ffn_output

        return y
