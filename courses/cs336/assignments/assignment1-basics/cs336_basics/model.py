import math

import torch
from torch import nn

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

        self.weigth = nn.Parameter(
            torch.empty(
                vocab_size,
                d_model,
                device = device,
                dtype = dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weigth,
            mean = 0.0,
            std = 1.0,
            a = -3.0,
            b = 3.0,
        )

    def forward(self, token_ids : torch.Tensor):
        return self.weigth[token_ids]

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
