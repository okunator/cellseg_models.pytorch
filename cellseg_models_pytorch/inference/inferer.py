from functools import partial
from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils.multiproc import run_pool, set_pool

from .post_processor import PostProcessor

__all__ = ["Inferer"]


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix.

    The matrix assigns bigger weight on pixels in the center and less weight to pixels
    on the image boundary. Helps dealing with prediction artifacts on tile boundaries.

    Ported from: pytorch-toolbelt

    Parameters:
        width (int):
            Tile width.
        height (int):
            Tile height.

    Returns:
        np.ndarray:
            Weight matrix. Shape (H, W).
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


class Inferer:
    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        post_proc_method: str,
        num_post_proc_threads: int = -1,
        mixed_precision: bool = False,
        **post_proc_kwargs,
    ) -> None:
        """The Inferer class is responsible for running inference and post-processing
        on a given segmentation model. It handles model predictions, applies activation
        functions, boundary weights, and post-processing of model predictions.

        Parameters:
            model (torch.nn.Module):
                The neural network model to be used for inference.
            input_shape (Tuple[int, int]):
                The shape of the input data (height, width).
            out_activations (Dict[str, str]):
                Dictionary specifying the activation functions for the output heads.
            out_boundary_weights (Dict[str, bool]):
                Dictionary specifying whether boundary weights should be used for each
                output head.
            post_proc_method (str):
                The method to be used for post-processing.
            num_post_proc_threads (int, default=-1):
                Number of threads to be used for post-processing.
                If -1, all available CPUs are used.
            mixed_precision (bool, default=False):
                Whether to use mixed precision during inference.
            **post_proc_kwargs:
                Additional keyword arguments for the PostProcessor instance.

        Raises:
            ValueError: If an invalid activation is specified in out_activations.

        Attributes:
            model (torch.nn.Module): The neural network model.
            pool (ThreadPool): Thread pool for post-processing.
            weight_mat (torch.Tensor, optional): Boundary weight matrix.
            post_processor (PostProcessor): Post-processor instance.

        Methods:
            predict(x: torch.Tensor, output_shape: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
            Run model prediction on the input tensor and return the model predictions.

            post_process(probs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
            Post-process the model predictions and return the post-processed predictions.

            post_process_parallel(probs: Dict[str, torch.Tensor], maptype: str = "amap") -> List[Dict[str, np.ndarray]]:
            Run the full post-processing pipeline in parallel for many model outputs and return the post-processed outputs.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.mixed_precision = mixed_precision
        self.out_activations = out_activations
        self.out_boundary_weights = out_boundary_weights
        self.post_proc_method = post_proc_method
        self.num_post_proc_threads = num_post_proc_threads
        self.out_heads = self._get_out_info()  # the names and num channels of out heads
        self.head_kwargs = self._check_and_set_head_args()
        self.pool = set_pool("thread", num_post_proc_threads)

        # check the out activations
        allowed_acts = ("softmax", "sigmoid", "tanh", None)
        for key, value in self.out_activations.items():
            if value not in allowed_acts:
                raise ValueError(
                    f"Invalid activation function: '{value}' for key: '{key}'. "
                    "Allowed activations: 'softmax', 'sigmoid', 'tanh', None"
                )

        # create the boundary weight matrix
        if any(self.out_boundary_weights.values()):
            weight_mat = compute_pyramid_patch_weight_loss(*input_shape)
            self.weight_mat = (
                torch.from_numpy(weight_mat)
                .float()
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # initialize the post-processor
        self.post_processor = PostProcessor(
            method=self.post_proc_method,
            inst_key=self.model.inst_key,
            aux_key=self.model.aux_key,
            **post_proc_kwargs,
        )

    def predict(
        self,
        x: torch.Tensor,
        output_shape: Tuple[int, int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the input through the model.

        Note:
            No post-processing is applied.

        Parameters:
            x torch.Tensor:
                Input tensor. Shape (N, C, H, W).
            output_shape (Tuple[int, int], default=None):
                Interpolate the model outputs to this shape.

        Returns:
            Dict[str, torch.Tensor]:
                Dictionary containing the model predictions (probabilities).
        """
        with torch.no_grad():
            if self.mixed_precision:
                with torch.autocast(self.device.type, dtype=torch.float16):
                    probs = self._predict(x, output_shape=output_shape)
            else:
                probs = self._predict(x, output_shape=output_shape)

        return probs

    def post_process(
        self,
        probs: Dict[str, torch.Tensor],
        save_path: Union[List[str], str] = None,
        maptype: str = "amap",
        save_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Post-process the model predictions.

        Parameters:
            probs (Dict[str, torch.Tensor]):
                Dictionary containing the model predictions. Shapes: (B, C, H, W).
                E.g. {"inst": tensor, "aux": tensor, "type": tensor, "sem": tensor}
            save_path (Union[List[str], str], default=None):
                Path to save the post-processed predictions. If None, predictions are not
                saved to disk but returned as a dictionary. If the batch size is greater
                than 1, the save_path should be a list of paths.
            maptype (str):
                The map type for pathos ThreadPool object for parallel post-processing.
                Ignored if `probs` batch size=1. Allowed: ("map","amap","imap","uimap")
            save_kwargs (Dict[str, Any], default=None):
                Keyword arguments for saving the post-processed predictions.
                See `FileHandler.to_mat` and `FileHandler.to_gson` for more details.
            **kwargs:
                Additional keyword arguments for the post-processing pipeline.

        Raises:
            ValueError: If the input shape is invalid.


        Returns:
            Dict[str, torch.Tensor]:
                Dictionary containing the post-processed model predictions.
        """
        if save_kwargs is None:
            save_kwargs = {}

        if probs[self.model.inst_key].ndim != 4:
            raise ValueError(
                "Invalid input shape. Expected `probs` to be 4D tensors. "
                f"Got: {[(k, v.shape) for k, v in probs.items()]}"
            )

        if probs[self.model.inst_key].shape[0] == 1:
            arg_order = [self.model.inst_key, self.model.aux_key, "type", "sem", "cyto"]
            inputs = [
                self._to_ndarray(probs[arg]) for arg in arg_order if arg in probs.keys()
            ]
        else:
            return self._post_process_parallel(
                probs,
                save_paths=save_path,
                maptype=maptype,
                save_kwargs=save_kwargs,
                **kwargs,
            )

        return self.post_processor.post_proc_pipeline(
            *inputs, save_path=save_path, save_kwargs=save_kwargs, **kwargs
        )

    def _post_process_parallel(
        self,
        probs: Dict[str, torch.Tensor],
        save_paths: List[str] = None,
        save_kwargs: Dict[str, Any] = None,
        maptype: str = "amap",
    ) -> List[Dict[str, np.ndarray]]:
        """Run the full post-processing pipeline in parallel for many model outputs.

        Parameters:
            probs (Dict[str, torch.Tensor]):
                Dictionary containing the model predictions. Shapes (B, C, H, W).
                E.g. {"inst": tensor, "aux": tensor, "type": tensor, "sem": tensor}
            save_paths (List[str], default=None):
                Paths to save the post-processed predictions. If None, predictions are not
            maptype (str, default="amap"):
                The map type of the pathos Pool object.
                Allowed: ("map", "amap", "imap", "uimap")

        Returns:
            List[Dict[str, np.ndarray]]:
                The post-processed output map dictionaries in a list.
        """
        pp = partial(self.post_processor.post_proc_pipeline, save_kwargs=save_kwargs)

        def _post_proc_func(inputs):
            if len(inputs) == 2:
                inputs, save_path = inputs
            return pp(*inputs, save_path=save_path)

        # zip the model outputs to tuple args in the order of `arg_order`
        arg_order = [self.model.inst_key, self.model.aux_key, "type", "sem", "cyto"]
        inputs = zip(
            *[
                [self._to_ndarray(m) for m in list(probs[arg])]
                for arg in arg_order
                if arg in probs.keys()
            ]
        )

        if save_paths is not None:
            run_pool(
                self.pool,
                _post_proc_func,
                list(zip(inputs, save_paths)),
                ret=False,
                maptype=maptype,
            )

        return run_pool(
            self.pool,
            _post_proc_func,
            inputs,
            ret=True,
            maptype=maptype,
        )

    def _predict(
        self,
        x: torch.Tensor,
        output_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Model predict pipeline."""
        # logits
        logits = self.model(x)

        probs = {}
        for k, out in logits.items():
            apply_boundary_weights = self.head_kwargs[k]["apply_weights"]
            activation = self.head_kwargs[k]["act"]

            if apply_boundary_weights:
                out *= self.weight_mat

            if activation == "softmax":
                out = torch.softmax(out, dim=1)
            elif activation == "sigmoid":
                out = torch.sigmoid(out)
            elif activation == "tanh":
                out = torch.tanh(out)

            # interpolate to the `output_shape`
            if output_shape is not None:
                out = F.interpolate(
                    out, size=output_shape, mode="bilinear", align_corners=False
                )

            probs[k] = out

        return probs

    def _to_ndarray(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.is_cuda:
            tensor = tensor.cpu()

        if tensor.shape[0] == 1:
            return tensor.squeeze().numpy()

        return tensor.numpy()

    def _check_and_set_head_args(self) -> None:
        """Check the model has matching head names with the head args and set them."""
        boundary_keys = self.out_boundary_weights.keys()
        act_keys = self.out_activations.keys()
        if boundary_keys != act_keys:
            raise ValueError(
                "Mismatching head names in `out_boundary_weights` & `out_activations`. "
                f"Got: {list(boundary_keys)} & {list(act_keys)}",
            )

        pred_kwargs = {}
        heads = [head_name for head_name, _ in self.out_heads]
        for head_name, val in self.out_activations.items():
            if head_name in heads:
                pred_kwargs[head_name] = {"act": val}
                pred_kwargs[head_name]["apply_weights"] = self.out_boundary_weights[
                    head_name
                ]
            else:
                raise ValueError(
                    f"Mismatching head name. The model contains heads: {heads}. "
                    f"Got: {head_name}",
                )

        return pred_kwargs

    def _get_out_info(self) -> Tuple[Tuple[str, int]]:
        """Get the output names and number of out channels."""
        return tuple(
            chain.from_iterable(
                list(self.model.heads[k].items()) for k in self.model.heads.keys()
            )
        )
