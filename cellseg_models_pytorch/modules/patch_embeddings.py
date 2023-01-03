from typing import Any, Dict

import torch
import torch.nn as nn

from .base_modules import Norm

__all__ = ["ContiguousEmbed", "PatchEmbed"]


class ContiguousEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 1,
        stride: int = 1,
        kernel_size: int = None,
        pad: int = 0,
        head_dim: int = 64,
        num_heads: int = 8,
        flatten: bool = True,
        normalization: str = None,
        norm_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Patch an image with nn.Conv2d and then embed.

        NOTE:
        The input is patched via nn.Conv2d i.e the patch dimensions are defined by the
        convolution parameters. The default values are set such that every pixel value
        is a patch. For big inputs this results in OOM errors when computing attention.

        If there is a need for bigger patches with no overlap, you can set for example
        `patch_size = 16` and `stride = 16` to get patches of size 16**2.

        - Input shape: (B, C, H, W)
        - Output shape: (B, H'*W', head_dim*num_heads)
            (If `patch_size=1` & `stride=1` -> H'*W'=H*W).

        NOTE: Optional normalization of the input before patching and projecting.

        Parameters
        ----------
            in_channels : int
                Number of input channels in the input tensor. (3 for RGB).
            patch_size : int, default=1
                Size of the patch. Defaults to 1, meaning that every pixel is a patch.
                (Given that stride is equal to 1.) If `kernel_size` is given, this will
                be ignored.
            stride : int, default=1
                The sliding window stride. Defaults to 1, meaning that every pixel is a
                patch. (Given that patch_size is equal to 1).
            kernel_size : int, optional
                The kernel size for the convolution. If None, the `patch_size` is used.
            pad : int, default=0
                Size of the padding.
            head_dim : int, default=64
                Number of channels per each head.
            num_heads : int, default=8
                Number of heads in multi-head self-attention.
            flatten : bool, default=True
                If True, the output will be flattened to a sequence. After flattening
                output will have shape (B, H'*W', head_dim*num_heads). If False, the
                output shape will remain (B, C, H', W').
            normalization : str, optional
                The name of the normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            **norm_kwargs : Dict[str, Any]
                key-word args for the normalization layer. Ignored if normalization
                is None.

        Examples
        --------
            >>> x = torch.rand([1, 3, 256, 256])

            >>> # per-pixel patches of shape 256*256
            >>> conv_patch = ContiguousEmbed(
                    in_channels=3,
                    patch_size=1,
                    stride=1,
                )
            >>> print(conv_patch(x).shape)
            >>> # torch.Size([1, 65536, 512])

            >>> # 16*16 patches
            >>> conv_patch2 = ContiguousEmbed(
                    in_channels=3,
                    patch_size=16,
                    stride=16,
                )
            >>> print(conv_patch2(x).shape)
            >>> # torch.Size([1, 256, 512])

            >>> # Downsampling input to patches of shape 64*64
            >>> conv_patch3 = ContiguousEmbed(
                    in_channels=3,
                    stride=4,
                    kernel_size=7,
                    pad=2
                )
            >>> print(conv_patch3(x).shape)
            >>> # torch.Size([1, 4096, 512])
        """
        super().__init__()
        self.flatten = flatten
        self.proj_dim = head_dim * num_heads
        self.kernel_size = patch_size if kernel_size is None else kernel_size
        self.pad = pad
        self.stride = stride
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.norm = Norm(normalization, **norm_kwargs)
        self.proj = nn.Conv2d(
            in_channels,
            self.proj_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad,
        )

    def get_patch_size(self, img_size: int) -> int:
        """Get the patch size from the conv params."""
        return int(
            (((img_size + 2 * self.pad - (self.kernel_size - 1)) - 1) / self.stride) + 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for projection."""
        B, _, H, W = x.shape

        # 1. Normalize
        x = self.norm(x)

        # 2. Patch and project.
        x = self.proj(x)  # (B, proj_dim, H', W')

        # 3. reshape to a sequence.
        # Every patch has been projected into a `proj_dim` long vector.
        if self.flatten:
            p_H = self.get_patch_size(H)
            p_W = self.get_patch_size(W)

            # flatten
            x = x.permute(0, 2, 3, 1).reshape(
                B, p_H * p_W, self.proj_dim
            )  # (B, H'*W', proj_dim)

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 16,
        head_dim: int = 64,
        num_heads: int = 8,
        normalization: int = None,
        **norm_kwargs,
    ) -> None:
        """Patch an input image and then embed/project.

        NOTE: This implementation first patches the input image by reshaping it
        and then embeds/projects it with nn.Linear.

        NOTE: Optional normalization of the input before patching and projecting.

        - Input shape: (B, C, H, W)
        - Patched shape: (B, H//patch_size * W//patch_size, C*patch_size**2)
        - Embedded output shape: (B, H//patch_size * W//patch_size, head_dim*num_heads)

        Parameters
        ----------
            in_channels : int
                Number of input channels in the input tensor.
            patch_size : int, default=16
                The H and W size of the patch.
            head_dim : int, default=64
                Number of channels per each head.
            num_heads : int, default=8
                Number of heads in multi-head self-attention.
            normalization : str, optional
                The name of the normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            **norm_kwargs : Dict[str, Any]
                key-word args for the normalization layer. Ignored if normalization
                is None.

        Examples
        --------
            >>> x = torch.rand([1, 3, 256, 256])

            >>> # patches of shape 16*16
            >>> lin_patch = PatchEmbed(
                    in_channels=3,
                    patch_size=16,
                )
            >>> print(lin_patch(x).shape)
            >>> # torch.Size([1, 256, 512])

        """
        super().__init__()
        self.proj_dim = head_dim * num_heads
        self.patch_size = patch_size
        self.norm = Norm(normalization, **norm_kwargs)
        self.proj = nn.Linear(in_channels * (patch_size**2), self.proj_dim)

    def img2patch(self, x: torch.Tensor) -> torch.Tensor:
        """Patch an input image of shape (B, C, H, W).

        Adapted from: PyTorch Lightning ViT tutorial.

        Parameters
        ----------
            x : torch.Tensor
                Input image of shape (B, C, H, W).

        Returns
        -------
            torch.Tensor:
                Patched and flattened input image.
                Shape: (B, H//patch_size * W//patch_size, C*patch_size**2)
        """
        B, C, H, W = x.shape
        x = x.reshape(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )  # (B, C, H', patch_size, W', patch_size)

        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H', W', C, p_H, p_W)
        x = x.flatten(1, 2)  # (B, H'*W', C, p_H, p_W)
        x = x.flatten(2, 4)  # (B, H'*W', C*p_H*p_W)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward patch embedding."""
        # 1. Normalize
        x = self.norm(x)  # (B, C, H, W)

        # 2. Patch
        x = self.img2patch(x)  # (B, H//patch_size * W//patch_size, C*patch_size**2)

        # 3. Project/Embed
        x = self.proj(x)

        return x
