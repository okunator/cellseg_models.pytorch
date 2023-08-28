from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["BaseTrEncoder"]


class BaseTrEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        out_indices: Optional[Tuple[int, ...]] = None,
        depth: int = 4,
        **kwargs,
    ) -> None:
        """Create a base class for all transformer encoders.

        Parameters
        ----------
        name : str
            Name of the backbone.
        checkpoint_path : Optional[Union[Path, str]], optional
            Path to the weights of the backbone. If None, the backbone is initialized
            with random weights. Defaults to None.
        out_indices : Tuple[int, ...], optional
            The indices of the features to be returned from the backbone. If None, the
            default is set to the last `depth` features of the backbone. Defaults to
            None.
        depth : int, default=4
            The number of selected features to be returned from the backbone. The
            features will be the last `depth` features of the backbone. Defaults to 4.
        """
        super().__init__()
        self.name = name
        self.depth = depth

        # set checkpoint path
        self.ckpt_path = checkpoint_path
        if checkpoint_path is not None:
            if checkpoint_path.startswith("https://"):
                self.is_url = True
                self.ckpt_path = checkpoint_path
            else:
                self.is_url = False
                self.ckpt_path = Path(checkpoint_path)

        self._out_inds = out_indices

    @property
    def out_indices(self):
        """Get the indices of the output features."""
        if self._out_inds is None:
            return tuple(
                range(
                    self.backbone.n_blocks - self.depth,
                    self.backbone.n_blocks,
                )
            )
        else:
            return self._out_inds

    def _strip_state_dict(self, state_dict: Dict) -> Dict:
        """Overload this method to strip the unnecessary parts of the state dict."""
        raise NotImplementedError

    def load_checkpoint(self) -> None:
        """Load the weights from the checkpoint."""
        backbone = self.backbone
        if self.ckpt_path is not None:
            if self.is_url:
                state_dict = torch.hub.load_state_dict_from_url(self.ckpt_path)
            else:
                state_dict = torch.load(
                    self.ckpt_path, map_location=lambda storage, loc: storage
                )
            try:
                msg = backbone.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                new_ckpt = self._strip_state_dict(state_dict)
                msg = backbone.load_state_dict(new_ckpt, strict=True)
            except BaseException as e:
                raise RuntimeError(f"Error loading checkpoint: {e}")

            print(f"Loading pre-trained {self.name} checkpoint: {msg}")
        self.backbone = backbone
