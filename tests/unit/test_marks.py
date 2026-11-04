"""marks.py: "simulation" written into a picture's own metadata (JPEG comment, PNG text chunk)."""
import cv2
import numpy as np
import pytest

from mimic import marks


def picture(path):
    image = np.full((40, 60, 3), 200, np.uint8)
    cv2.rectangle(image, (5, 5), (30, 30), (0, 0, 255), -1)
    cv2.imwrite(str(path), image)
    return image


def test_a_jpeg_keeps_its_picture_and_gains_a_comment(tmp_path):
    path = tmp_path / "a.jpg"
    picture(path)
    before = cv2.imread(str(path))
    marks.mark_jpeg(path, "SIMULATION: test")
    marks.mark_jpeg(path, "SIMULATION: test")            # twice: still one comment
    assert marks.jpeg_comment(path) == "SIMULATION: test"
    assert path.read_bytes().count(b"SIMULATION: test") == 1
    assert (cv2.imread(str(path)) == before).all()


def test_a_png_keeps_its_picture_and_gains_a_text_chunk(tmp_path):
    path = tmp_path / "a.png"
    image = picture(path)
    marks.mark_png(path, "Comment", "SIMULATION: test")
    assert marks.png_texts(path) == {"Comment": "SIMULATION: test"}
    assert (cv2.imread(str(path)) == image).all()


def test_the_wrong_kind_of_file_is_refused(tmp_path):
    path = tmp_path / "a.png"
    picture(path)
    with pytest.raises(ValueError, match="not a JPEG"):
        marks.mark_jpeg(path, "x")
