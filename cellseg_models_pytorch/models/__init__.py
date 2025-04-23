# import torch.nn as nn

# from .cellpose.cellpose import (
#     CellPoseUnet,
#     cellpose_base,
#     cellpose_plus,
#     omnipose_base,
#     omnipose_plus,
# )
# from .cellvit.cellvit import (
#     CellVitSAM,
#     cellvit_sam_base,
#     cellvit_sam_plus,
#     cellvit_sam_small,
#     cellvit_sam_small_plus,
# )
# from .cppnet.cppnet import CPPNet, cppnet_base, cppnet_base_multiclass, cppnet_plus
# from .hovernet.hovernet import (
#     HoverNet,
#     hovernet_base,
#     hovernet_plus,
#     hovernet_small,
#     hovernet_small_plus,
# )
# from .stardist.stardist import (
#     StarDistUnet,
#     stardist_base,
#     stardist_base_multiclass,
#     stardist_plus,
# )

# MODEL_LOOKUP = {
#     "cellpose_base": cellpose_base,
#     "cellpose_plus": cellpose_plus,
#     "omnipose_base": omnipose_base,
#     "omnipose_plus": omnipose_plus,
#     "hovernet_base": hovernet_base,
#     "hovernet_plus": hovernet_plus,
#     "hovernet_small": hovernet_small,
#     "hovernet_small_plus": hovernet_small_plus,
#     "stardist_base": stardist_base,
#     "stardist_plus": stardist_plus,
#     "stardist_base_multiclass": stardist_base_multiclass,
#     "cellvit_sam_base": cellvit_sam_base,
#     "cellvit_sam_plus": cellvit_sam_plus,
#     "cellvit_sam_small": cellvit_sam_small,
#     "cellvit_sam_small_plus": cellvit_sam_small_plus,
#     "cppnet_base": cppnet_base,
#     "cppnet_base_multiclass": cppnet_base_multiclass,
#     "cppnet_plus": cppnet_plus,
# }


# def get_model(
#     name: str,
#     type: str,
#     n_type_classes: int = None,
#     n_sem_classes: int = None,
#     **kwargs,
# ) -> nn.Module:
#     """Get the corect model at hand given name and type.

#     Parameters:
#         name (str):
#             Name of the model.
#         type (str):
#             Type of the model. One of "base", "plus", "small", "small_plus".
#         n_type_classes (int):
#             Number of cell types to segment.
#         n_sem_classes (int):
#             Number of tissue types to segment.
#         **kwargs
#             Additional keyword arguments.

#     Returns:
#         nn.Module: The specified model.
#     """
#     if name == "stardist":
#         if type == "base":
#             model = MODEL_LOOKUP["stardist_base_multiclass"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["stardist_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     elif name == "cppnet":
#         if type == "base":
#             model = MODEL_LOOKUP["cppnet_base_multiclass"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["cppnet_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     elif name == "cellpose":
#         if type == "base":
#             model = MODEL_LOOKUP["cellpose_base"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["cellpose_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     elif name == "omnipose":
#         if type == "base":
#             model = MODEL_LOOKUP["omnipose_base"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["omnipose_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     elif name == "hovernet":
#         if type == "base":
#             model = MODEL_LOOKUP["hovernet_base"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "small":
#             model = MODEL_LOOKUP["hovernet_small"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["hovernet_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#         elif type == "small_plus":
#             model = MODEL_LOOKUP["hovernet_small_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     elif name == "cellvit":
#         if type == "base":
#             model = MODEL_LOOKUP["cellvit_sam_base"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "small":
#             model = MODEL_LOOKUP["cellvit_sam_small"](
#                 n_type_classes=n_type_classes, **kwargs
#             )
#         elif type == "plus":
#             model = MODEL_LOOKUP["cellvit_sam_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#         elif type == "small_plus":
#             model = MODEL_LOOKUP["cellvit_sam_small_plus"](
#                 n_type_classes=n_type_classes, n_sem_classes=n_sem_classes, **kwargs
#             )
#     else:
#         raise ValueError("Unknown model type or name.")

#     return model


# __all__ = [
#     "HoverNet",
#     "hovernet_base",
#     "hovernet_plus",
#     "hovernet_small",
#     "hovernet_small_plus",
#     "CellPoseUnet",
#     "cellpose_base",
#     "cellpose_plus",
#     "omnipose_base",
#     "omnipose_plus",
#     "StarDistUnet",
#     "stardist_base",
#     "stardist_plus",
#     "stardist_base_multiclass",
#     "MODEL_LOOKUP",
#     "get_model",
#     "CellVitSAM",
#     "cellvit_sam_base",
#     "cellvit_sam_plus",
#     "cellvit_sam_small",
#     "cellvit_sam_small_plus",
#     "cppnet_base",
#     "cppnet_base_multiclass",
#     "cppnet_plus",
#     "CPPNet",
# ]
