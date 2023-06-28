"""
Raw image definition and related functions.
"""

from dataclasses import dataclass, field
import pickle

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class RawImage:
    """
    A raw image, represented as an array of grayscale values.
    """
    width: int
    height: int
    data: np.ndarray

    def __post_init__(self):
        assert self.data.shape == (self.height, self.width)
    

def import_image_from_file(path: str) -> RawImage:
    assert path.endswith(".pkl")
    with open(path, "rb") as fh:
        image = pickle.load(fh)
        return image


def import_image_from_jpeg(path: str) -> RawImage:
    """
    Import a JPEG image from disk.
    """
    assert path.endswith(".jpg") or path.endswith(".jpeg")
    data = plt.imread(path, format="jpeg")

    # Convert to grayscale
    data = np.mean(data, axis=(2))

    height, width = data.shape
    return RawImage(width, height, data)



def export_image_to_file(image: RawImage, path: str):
    assert path.endswith(".pkl")
    with open(path, "wb") as fh:
        pickle.dump(image, fh)

def random_gradient(width: int, height: int) -> RawImage:
    """
    Generate a random gradient image.
    """
    unordered = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    ordered = np.sort(unordered, axis=0)
    ordered = np.sort(ordered, axis=1)
    return RawImage(width, height, ordered)
