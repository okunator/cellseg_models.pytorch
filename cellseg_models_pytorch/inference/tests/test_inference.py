# import pytest

# from cellseg_models_pytorch.inference import Inferer, SlidingWindowInferer
# from cellseg_models_pytorch.models import cellpose_plus
# import numpy as np
# import torch

# def test_slidingwin_inferer(img_sample512):
#     model = cellpose_plus(n_sem_classes=3, n_type_classes=3, long_skip="unet")

#     inferer = SlidingWindowInferer(
#         model,
#         patch_shape=(256, 256),
#         stride=256,
#         out_activations={"sem-sem": "softmax", "cellpose-type": "softmax", "cellpose-cellpose": "tanh"},
#         out_boundary_weights={"sem-sem": False, "cellpose-type": False, "cellpose-cellpose": True},
#         post_proc_method="cellpose",
#         num_post_proc_threads=1,
#         mixed_precision=True,
#     )

#     im = img_sample512
#     im = np.transpose(im, (2, 0, 1))  
#     im = np.expand_dims(im, axis=0)  
#     im = torch.tensor(im)
#     probs = inferer.predict(im.float())
#     out_masks = inferer.post_process(probs)

#     assert out_masks["sem-sem"].shape == (256, 256)


# def test_inferer(img_sample256):
#     model = cellpose_plus(n_sem_classes=3, n_type_classes=3, long_skip="unet")

#     inferer = Inferer(
#         model,
#         input_shape=(256, 256),
#         out_activations={"sem-sem": "softmax", "cellpose-type": "softmax", "cellpose-cellpose": "tanh"},
#         out_boundary_weights={"sem-sem": False, "cellpose-type": False, "cellpose-cellpose": True},
#         post_proc_method="cellpose",
#         num_post_proc_threads=1,
#         mixed_precision=True,
#     )

#     im = img_sample256
#     im = np.transpose(im, (2, 0, 1))  
#     im = np.expand_dims(im, axis=0)  
#     im = torch.tensor(im)
#     probs = inferer.predict(im.float())
#     out_masks = inferer.post_process(probs)

#     assert out_masks["sem-sem"].shape == (256, 256)
