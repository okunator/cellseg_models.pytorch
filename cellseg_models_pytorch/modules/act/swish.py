import torch
import torch.nn as nn

__all__ = ["Swish"]


@torch.jit.script
def swish_jit_fwd(input: torch.Tensor) -> torch.Tensor:
    """Swish forward."""
    return input.mul(torch.sigmoid(input))


@torch.jit.script
def swish_jit_bwd(input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Swish backward."""
    input_s = torch.sigmoid(input)
    return grad_output * (input_s * (1 + input * (1 - input_s)))


class SwishFunction(torch.autograd.Function):
    """Memory efficient Swish implementation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass."""
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Apply element-wise swish function."""
    return SwishFunction.apply(x)


class Swish(nn.Module):
    def __init__(self, inplace: bool = False, **kwargs) -> None:
        """Apply the element-wise swish function.

        Parameters
        ----------
            inplace : bool, default=False
                This is not used, exists only for compatibility.
        """
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of swish activation.

        Parameters
        ----------
            input : torch.Tensor
                Input tensor. Can be of any shape (C, *)

        Returns
        -------
            torch.Tensor:
                Activated output tensor. Shape same as input.
        """
        return swish(input)
