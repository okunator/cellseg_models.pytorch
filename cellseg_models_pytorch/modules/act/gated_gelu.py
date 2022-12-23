"""
Adapted from the diffusers repo. (Added type hints).

https://github.com/huggingface/diffusers/blob/v0.7.0/src/diffusers/models/attention.py

Copyright [2022] [huggingface]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GEGLU", "ApproximateGELU"]


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, **kwargs) -> None:
        """Apply a variant of the gated linear unit activation function.

        https://arxiv.org/abs/2002.05202.

        Parameters
        ----------
            dim_in : int
                The number of channels in the input.
            dim_out : int
                The number of channels in the output.
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        """Run gelu activation."""
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of he gated GELU."""
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gelu(gate)


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, **kwargs) -> None:
        """Apply the approximate form of Gaussian Error Linear Unit (GELU).

        https://arxiv.org/abs/1606.08415

        Parameters
        ----------
            dim_in : int
                The number of channels in the input.
            dim_out : int
                The number of channels in the output.
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the approximate GELU."""
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
