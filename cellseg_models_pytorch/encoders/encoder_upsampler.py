from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["EncoderUpsampler", "FeatUpsampleBlock"]


class FeatUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        scale_factor: int = 2,
    ) -> None:
        """Upsample 2D dimensions of a feature.

        TransConv + Conv layers

        Parameters:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            scale_factor (int):
                Scale factor for upsampling. Defaults to 2.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.scale_factor = scale_factor
        self.out_channels = out_channels

        if isinstance(scale_factor, int):
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 ** (scale_factor - 1),
                stride=2 ** (scale_factor - 1),
                padding=0,
                output_padding=0,
            )
        else:
            self.up = nn.Upsample(
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=True,
            )

        self.conv_block = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_block(x)
        return x


class EncoderUpsampler(nn.Module):
    def __init__(
        self,
        feature_info: Tuple[dict, ...],
        out_channels: Tuple[int, ...],
    ) -> None:
        """Feature upsampler for transformer-like backbones.

        Note:
            This is a U-NetTR like upsampler that takes the features from the backbone
            and upsamples them such that the scale factor between the upsampled features
            are two. Builds an image-pyramid like structure.

        Parameters:
            feature_info (Tuple[dict, ...]):
                timm feature info of the backbone. Assumes that the feature info dicts
                are in bottleneck first order I.e. the deepest encoder block first.
                For example: [
                    {'module': 'blocks.8', 'num_chs': 1024, 'reduction': 16},
                    {'module': 'blocks.4', 'num_chs': 1024, 'reduction': 16}
                }
            out_channels (Tuple[int, ...]):
                Number of channels in the output tensor of each upsampling block.
                Defaults to None.
        """
        super().__init__()
        if len(out_channels) != len(feature_info):
            raise ValueError(
                "`out_channels` must have the same len as the `backbone.feature_info.`"
                f"Got {len(out_channels)} and {len(feature_info)} respectively."
            )

        self.out_channels = out_channels
        self.feature_info = []

        # bottleneck layer
        self.bottleneck = nn.Conv2d(
            in_channels=feature_info[0]["num_chs"],
            out_channels=self.out_channels[0],
            kernel_size=1,
        )

        # add timm-like feature info of the bottleneck layer
        self.feature_info.append(
            {
                "num_chs": self.out_channels[0],
                "module": "bottleneck",
                "reduction": float(feature_info[0]["reduction"]),
            }
        )

        self.up_blocks = nn.ModuleDict()
        n_up_blocks = list(range(1, len(self.out_channels)))
        for i, (out_chls, finfo, n_blocks) in enumerate(
            zip(self.out_channels[1:], feature_info[1:], n_up_blocks)
        ):
            up_blocks = []
            squeeze_rates = list(range(n_blocks))[::-1]

            for j, squeeze_ratio in zip(range(n_blocks), squeeze_rates):
                if j == 0:
                    in_channels = finfo["num_chs"]
                else:
                    in_channels = up.out_channels  # noqa

                up = FeatUpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_chls * (2**squeeze_ratio),
                    scale_factor=2,
                )
                up_blocks.append(up)

            # add feature info
            self.feature_info.append(
                {
                    "num_chs": out_chls,
                    "module": f"up{i + 1}",
                    "reduction": finfo["reduction"] / 2**n_blocks,
                }
            )
            self.up_blocks[f"up{i + 1}"] = nn.Sequential(*up_blocks)

    def forward(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the encoder upsampler.

        Parameters:
            feats (Tuple[torch.Tensor]):
                Tuple of features from the backbone in bottleneck first order. I.e. the
                bottleneck (deepest) feature is the first element in the tuple.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of upsampled features in hi-to-lo res order.
        """
        # bottleneck feature
        up_feat = self.bottleneck(feats[0])
        intermediate_features = [up_feat]

        # upsampled features
        for i, feat in enumerate(feats[1:]):
            up_feat = self.up_blocks[f"up{i + 1}"](feat)
            intermediate_features.append(up_feat)

        return tuple(intermediate_features)  # hi-to-lo res order
