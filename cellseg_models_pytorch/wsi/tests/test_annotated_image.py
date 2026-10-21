import numpy as np
import pytest
from PIL import Image

from cellseg_models_pytorch.wsi.image import get_annotated_image


@pytest.mark.parametrize("highlight", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_annotation_preserves_scaled_rectangles_blending_and_input(
    highlight: bool, alpha: float
) -> None:
    image = np.full((16, 18, 3), [20, 40, 60], dtype=np.uint8)
    original = image.copy()
    coordinates = [(4, 4, 8, 8), (20, 4, 8, 8)]

    result = get_annotated_image(
        image, coordinates, 2.0, highlight_first=highlight, alpha=alpha
    )

    expected = original.copy()
    first_color = [0, 0, 255] if highlight else [255, 0, 0]
    for left, color in [(2, first_color), (10, [255, 0, 0])]:
        expected[2, left : left + 5] = color
        expected[6, left : left + 5] = color
        expected[2:7, left] = color
        expected[2:7, left + 4] = color
    expected = ((1 - alpha) * expected + alpha * original).astype(np.uint8)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    assert coordinates == [(4, 4, 8, 8), (20, 4, 8, 8)]


def test_annotation_preserves_axis_scaling_fill_and_line_width() -> None:
    image = np.full((12, 10, 3), 255, dtype=np.uint8)

    result = get_annotated_image(
        image,
        [(4, 8, 24, 12)],
        (2.0, 4.0),
        rectangle_outline="blue",
        rectangle_fill="green",
        rectangle_width=2,
    )

    expected = image.copy()
    expected[4:11, 1:8] = [0, 0, 255]
    expected[6:9, 3:6] = [0, 128, 0]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("break_kind", ["none", "list", "tuple", "array"])
def test_annotation_preserves_continuous_colors_and_break_inputs(
    break_kind: str,
) -> None:
    image = np.full((12, 24, 3), 255, dtype=np.uint8)
    values = np.array([0.0, 1.0])
    breaks = None
    if break_kind == "list":
        breaks = [0.0, 1.0]
    elif break_kind == "tuple":
        breaks = (0.0, 1.0)
    elif break_kind == "array":
        breaks = np.array([0.0, 1.0])

    result = get_annotated_image(
        image,
        [(2, 2, 6, 6), (14, 2, 6, 6)],
        1.0,
        cmap="viridis",
        values=values,
        breaks=breaks,
    )

    pixels = np.asarray(result)
    np.testing.assert_array_equal(pixels[2, 2], [68, 1, 84])
    np.testing.assert_array_equal(pixels[2, 14], [253, 231, 36])
    np.testing.assert_array_equal(pixels[4, 4], [255, 255, 255])
    np.testing.assert_array_equal(values, [0.0, 1.0])
    if breaks is not None:
        np.testing.assert_array_equal(breaks, [0.0, 1.0])


@pytest.mark.parametrize("values", [np.array([20, 10]), np.array(["b", "a"])])
def test_annotation_preserves_categorical_colors_and_values(values: np.ndarray) -> None:
    image = np.full((12, 24, 3), 255, dtype=np.uint8)
    original = values.copy()

    result = get_annotated_image(
        image,
        [(2, 2, 6, 6), (14, 2, 6, 6)],
        1.0,
        cmap="viridis",
        values=values,
    )

    pixels = np.asarray(result)
    np.testing.assert_array_equal(pixels[2, 2], [253, 231, 36])
    np.testing.assert_array_equal(pixels[2, 14], [68, 1, 84])
    np.testing.assert_array_equal(values, original)


def test_annotation_draws_text_without_mutating_labels() -> None:
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    labels = ["A", "B"]

    result = get_annotated_image(
        image, [(4, 4, 56, 56), (70, 4, 56, 56)], 1.0, text_items=labels
    )

    pixels = np.asarray(result)
    assert np.any(np.all(pixels[6:58, 6:58] == 0, axis=-1))
    assert np.any(np.all(pixels[6:58, 72:124] == 0, axis=-1))
    assert labels == ["A", "B"]
    assert np.all(image == 255)


@pytest.mark.parametrize("gray", [False, True])
def test_annotation_with_no_coordinates_preserves_pixels(gray: bool) -> None:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    expected = np.repeat(image[..., None], 3, axis=-1)
    if not gray:
        image = expected.copy()

    result = get_annotated_image(image, [], 1.0)

    assert result.mode == "RGB"
    np.testing.assert_array_equal(result, expected)


def test_annotation_preserves_text_length_validation() -> None:
    with pytest.raises(ValueError, match="Length of text items"):
        get_annotated_image(
            np.zeros((8, 8, 3), np.uint8), [(0, 0, 4, 4)], 1.0, text_items=[]
        )


@pytest.mark.parametrize(
    "image",
    [np.zeros((8, 8), np.float32), np.zeros((8, 8, 4), np.uint8)],
)
def test_annotation_preserves_array_validation(image: np.ndarray) -> None:
    with pytest.raises(TypeError):
        get_annotated_image(image, [], 1.0)
