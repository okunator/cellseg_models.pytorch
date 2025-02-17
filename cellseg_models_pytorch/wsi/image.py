from typing import Optional, Union

import numpy as np

try:
    from matplotlib.font_manager import fontManager

    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False

from PIL import Image, ImageDraw, ImageFont

from .tiles import _divide_xywh
from .tissue import check_image

ERROR_TEXT_ITEM_LENGTH = (
    "Length of text items ({}) does not match length of coordinates ({})."
)


def get_annotated_image(
    image: Union[np.ndarray, Image.Image],
    coordinates: list[tuple[int, int, int, int]],
    downsample: Union[float, tuple[float, float]],
    *,
    rectangle_outline: str = "red",
    rectangle_fill: Optional[str] = None,
    rectangle_width: int = 1,
    highlight_first: bool = False,
    highlight_outline: str = "blue",
    text_items: Optional[list[str]] = None,
    text_color: str = "black",
    text_proportion: float = 0.75,
    text_font: str = "monospace",
    alpha: float = 0.0,
) -> Image.Image:
    """Function to draw tiles to an image. Useful for visualising tiles/predictions.

    Parameters:
        image (Union[np.ndarray, Image.Image]):
            Image to draw to.
        coordinates (list[tuple[int, int, int, int]]):
            Tile coordinates.
        downsample (Union[float, tuple[float, float]]):
            Downsample for the image. If coordinates are from the same image,
            set this to 1.0.
        rectangle_outline (str, default="red"):
            Outline color of each tile.
        rectangle_fill (Optional[str], default=None):
            Fill color of each tile..
        rectangle_width (int, default=1):
            Width of each tile edges.
        highlight_first (bool, default=False):
            Highlight first tile, useful when tiles overlap.
        highlight_outline (str, default="blue"):
            Highlight color for the first tile.
        text_items (Optional[list[str]], default=None):
            Text items for each tile. Length must match `coordinates`.
        text_color (str, default="black"):
            Text color.
        text_proportion (float, default=0.75):
            Proportion of space the text takes in each tile.
        text_font (str, default="monospace"):
            Passed to matplotlib's `fontManager.find_font` function.
        alpha (float, default=0.0):
            Alpha value for blending the original image and drawn image.

    Raises:
        ValueError: Text item length does not match length of coordinates.

    Returns:
        PIL.Image.Image:
            Annotated PIL image.
    """
    if not _has_matplotlib:
        raise ImportError(
            "Matplotlib is required for `get_annotated_image`. `pip install matplotlib`"
        )
    # Check image and convert to PIL Image.
    image = Image.fromarray(check_image(image)).convert("RGB")
    # Check arguments.
    if text_items is not None:
        if len(text_items) != len(coordinates):
            raise ValueError(
                ERROR_TEXT_ITEM_LENGTH.format(len(text_items), len(coordinates))
            )
    else:
        text_items = [None] * len(coordinates)
    # Draw tiles.
    font = None
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, (xywh, text) in enumerate(zip(coordinates, text_items)):
        # Downscale coordinates.
        x, y, w, h = _divide_xywh(xywh, downsample)
        # Draw rectangle.
        draw.rectangle(
            ((x, y), (x + w, y + h)),
            fill=rectangle_fill,
            outline=rectangle_outline,
            width=rectangle_width,
        )
        if text is not None:
            # Define font.
            if font is None:
                max_length = max(3, max(len(str(x)) for x in text_items))
                font_path = fontManager.findfont(text_font)
                # Figure width coefficient to find correct font size.
                font_32 = ImageFont.FreeTypeFont(font_path, size=32).getbbox("W")
                font_64 = ImageFont.FreeTypeFont(font_path, size=64).getbbox("W")
                font_coeff = font_64[2] / font_32[2]
                # Create font.
                font = ImageFont.FreeTypeFont(
                    font_path, size=round(font_coeff * text_proportion * w / max_length)
                )
            # Write text.
            draw.text(
                xy=(x + rectangle_width, y + rectangle_width),
                text=str(text_items[idx]),
                font=font,
                fill=text_color,
            )
    # Highlight first.
    if highlight_first and len(coordinates) > 0:
        x, y, w, h = _divide_xywh(coordinates[0], downsample)
        draw.rectangle(
            ((x, y), (x + w, y + h)),
            fill=rectangle_fill,
            outline=highlight_outline,
            width=rectangle_width,
        )
    # Blend.
    return Image.blend(annotated, image, alpha)
