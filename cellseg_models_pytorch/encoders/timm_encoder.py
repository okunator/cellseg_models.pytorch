from typing import List

import timm
import torch
import torch.nn as nn

__all__ = ["TimmEncoder"]


class TimmEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        checkpoint_path: str = None,
        in_channels: int = 3,
        depth: int = 4,
        out_indices: List[int] = None,
        **kwargs,
    ) -> None:
        """Import any encoder from timm package.

        Parameters
        ----------
            name : str
                Name of the encoder.
            pretrained : bool, optional
                If True, load pretrained weights, by default True.
            checkpoint_path : str, optional
                Path to the checkpoint file, by default None. If not None, overrides
                the `pretrained` argument.
            in_channels : int, optional
                Number of input channels, by default 3.
            depth : int, optional
                Number of output features, by default 4.
            out_indices : List[int], optional
                Indices of the output features, by default None. If None,
                out_indices is set to range(len(depth)). Overrides the `depth` argument.
            **kwargs : Dict[str, Any]
                Key-word arguments for any `timm` based encoder. These arguments are
                used in `timm.create_model(**kwargs)` function call.
        """
        super().__init__()

        # set out_indices
        self.out_indices = out_indices
        if out_indices is None:
            self.out_indices = tuple(range(depth))

        # set checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = ""

        # create the timm model
        try:
            self.backbone = timm.create_model(
                name,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                in_chans=in_channels,
                features_only=True,
                out_indices=self.out_indices,
                **kwargs,
            )
        except (AttributeError, RuntimeError) as err:
            print(err)
            raise RuntimeError(
                f"timm backbone: {name} is not supported due to missing "
                "features_only argument implementation in timm-package."
            )
        except IndexError as err:
            print(err)
            raise IndexError(
                f"It's possible that the given depth: {depth} is too large for "
                f"the given backbone: {name}. Try passing a smaller `depth` argument "
                "or a different backbone."
            )

        # set in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = tuple(self.backbone.feature_info.channels()[::-1])
        if out_indices is not None:
            self.out_channels = tuple(self.out_channels[i] for i in self.out_indices)

        self.feature_info = self.backbone.feature_info.info[:depth][::-1]

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """Forward pass of the encoder and return all the features."""
        features = self.backbone(x)
        return features[::-1]
