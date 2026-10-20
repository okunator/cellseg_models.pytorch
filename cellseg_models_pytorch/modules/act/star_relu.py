# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for this library; see NOTICE.md.

import torch
import torch.nn as nn

__all__ = ["StarReLU"]


class StarReLU(nn.Module):
    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        scale_learnable: bool = True,
        bias_learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        """Apply StarReLU activation.

        Adapted from:
        https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py

        See MetaFormer: https://arxiv.org/abs/2210.13452

        StarReLU: s * relu(x) ** 2 + b

        Args:
            scale_value: Learnable scaling factor for relu activation.
                Defaults to 1.0.
            bias_value: Learnable bias term for relu activation.
                Defaults to 0.0.
            scale_learnable: Flag, whether to keep the scale factor learnable.
                Defaults to True.
            bias_learnable: Flag, whether to keep the bias term learnable.
                Defaults to True.
            inplace: Flag whether to apply inplace-relu.
                Defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the StarReLU."""
        return self.scale * self.relu(x) ** 2 + self.bias
