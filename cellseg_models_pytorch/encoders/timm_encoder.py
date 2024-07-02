from typing import List, Tuple

import timm
import torch
import torch.nn as nn

__all__ = ["TimmEncoder"]


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_indices: tuple = None,
        extra_kwargs: dict = None,
    ) -> None:
        """Wrapper for timm models to output intermediate features.

        Note:
            The input timm model should have a `forward_intermediates` method.

        Parameters:
            model_name (str):
                Name of the timm model to use.
            pretrained (bool):
                Flag, whether to load pretrained weights.
            out_indices (tuple):
                Indices of the intermediate features to return.
            extra_kwargs (dict):
                Extra keyword arguments to pass to the timm model.
        """
        super().__init__()
        self.model_name = model_name

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            **(extra_kwargs or {}),
        )

        self.out_indices = out_indices
        if out_indices is None:
            self.out_indices = tuple(range(len(self.encoder.feature_info)))

        if len(self.out_indices) > 5:
            raise ValueError(
                f"Number of out_indices: {len(self.out_indices)} should be less than 5."
                "Specify the `out_indices` of the intermediate features to return"
                " to avoid this error."
            )

        if not hasattr(self.encoder, "forward_intermediates"):
            raise AttributeError(
                f"Model: {model_name} does not have a `forward_intermediates` method."
                " Please use a model that supports intermediate feature extraction."
            )

        self.feature_info = [self.encoder.feature_info[ind] for ind in self.out_indices]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        final_feat, intermediates = self.encoder.forward_intermediates(x)

        # HACK:
        # some timm models return different len intermediates and feature_info (convnext)
        # so we need to make sure that we return the correct intermediate features such
        # that the bottleneck featuremap is included
        offset = len(intermediates) - len(self.encoder.feature_info)

        return final_feat, [intermediates[i + offset] for i in self.out_indices]


# class TimmEncoder(nn.Module):
#     def __init__(
#         self,
#         name: str,
#         pretrained: bool = True,
#         checkpoint_path: str = None,
#         in_channels: int = 3,
#         depth: int = 4,
#         out_indices: List[int] = None,
#         **kwargs,
#     ) -> None:
#         """Import any encoder from timm package.

#         Parameters
#         ----------
#             name : str
#                 Name of the encoder.
#             pretrained : bool, optional
#                 If True, load pretrained weights, by default True.
#             checkpoint_path : str, optional
#                 Path to the checkpoint file, by default None. If not None, overrides
#                 the `pretrained` argument.
#             in_channels : int, optional
#                 Number of input channels, by default 3.
#             depth : int, optional
#                 Number of output features, by default 4.
#             out_indices : List[int], optional
#                 Indices of the output features, by default None. If None,
#                 out_indices is set to range(len(depth)). Overrides the `depth` argument.
#             **kwargs : Dict[str, Any]
#                 Key-word arguments for any `timm` based encoder. These arguments are
#                 used in `timm.create_model(**kwargs)` function call.
#         """
#         super().__init__()

#         # set out_indices
#         self.out_indices = out_indices
#         if out_indices is None:
#             self.out_indices = tuple(range(depth))

#         # set checkpoint_path
#         if checkpoint_path is None:
#             checkpoint_path = ""

#         # create the timm model
#         try:
#             self.backbone = timm.create_model(
#                 name,
#                 pretrained=pretrained,
#                 checkpoint_path=checkpoint_path,
#                 in_chans=in_channels,
#                 features_only=True,
#                 out_indices=self.out_indices,
#                 **kwargs,
#             )
#         except (AttributeError, RuntimeError) as err:
#             print(err)
#             raise RuntimeError(
#                 f"timm backbone: {name} is not supported due to missing "
#                 "features_only argument implementation in timm-package."
#             )
#         except IndexError as err:
#             print(err)
#             raise IndexError(
#                 f"It's possible that the given depth: {depth} is too large for "
#                 f"the given backbone: {name}. Try passing a smaller `depth` argument "
#                 "or a different backbone."
#             )

#         # set in_channels and out_channels
#         self.in_channels = in_channels
#         self.out_channels = tuple(self.backbone.feature_info.channels()[::-1])
#         if out_indices is not None:
#             self.out_channels = tuple(self.out_channels[i] for i in self.out_indices)

#         self.feature_info = self.backbone.feature_info.info[:depth][::-1]

#     def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
#         """Forward pass of the encoder and return all the features."""
#         features = self.backbone(x)
#         return features[::-1]
