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
