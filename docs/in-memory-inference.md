# In-memory sliding-window inference

Segment a large RGB image already loaded as a NumPy array. Overlapping windows
let the model process it at its original resolution, then combine its predictions
before reconstructing the nuclei.

For a worked example with an image and predicted masks, see
[Image inference](tutorials/inference.ipynb).

Set `image` to your array with shape `(height, width, 3)`. Choose trained
segmentation weights appropriate for your tissue. This example loads the
[HGSC HoVer-Net model](https://huggingface.co/csmp-hub/hovernet-histo-hgsc-nuc-v1),
which expects global min-max normalization. The weights download on first use.

```python
import torch

from cellseg_models_pytorch.models.hovernet import HoverNet
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HoverNet.from_pretrained("hgsc_v1_efficientnet_b5", device=device)
model.set_inference_mode(mixed_precision=False)

# Normalize the whole image once, keeping the original array unchanged.
normalized_image = minmax_normalize(image, copy=True)

prediction = model.predict(
    normalized_image,
    use_sliding_win=True,
    window_size=(256, 256),
    stride=128,
)

# Reconstruct nuclei after combining the overlapping window predictions.
result = model.post_process(prediction, use_async_postproc=False)
instances, _ = result["nuc"][0]
```

`instances` has the same height and width as `image`. Zero is background and each
positive integer identifies one nucleus. The model also predicts cell types;
the example discards that class mask when unpacking the result.

`stride=128` gives 50% overlap between 256-pixel windows. Choose a stride no larger
than either window dimension. Padding and cropping preserve the original image
coordinates, including images smaller than a window.

Use sliding windows when a full-image forward pass exceeds GPU memory, or when
a model requires a fixed input size. This preserves nucleus size instead of
shrinking the image. Overlap can reduce crop-edge artifacts compared with separate,
non-overlapping patches, at the cost of extra computation.

This implementation reduces memory used by intermediate model activations. The
full image and combined maps must still fit on the model’s device; it does not
stream windows from CPU memory to the GPU. For whole slides, use the
[WSI example](tutorials/wsi.ipynb) to read and process tiles from the slide file.
