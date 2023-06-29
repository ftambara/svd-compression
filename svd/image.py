"""
Raw image definition and related functions.
"""

from dataclasses import dataclass, field
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


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


@dataclass
class SVDImage:
    """SVD factorization of a RawImage."""
    width: int
    height: int
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        assert self.u.shape == (self.height, self.height)
        assert self.s.shape == (self.height, self.width)
        assert self.v.shape == (self.width, self.width)

    @property
    def data(self) -> np.ndarray:
        return self.u @ self.s @ self.v
    
    @classmethod
    def from_raw_image(cls, image: RawImage) -> "SVDImage":
        u, s_vector, v = la.svd(image.data)
        # If matrix is rectangular, pad s with zeros
        s = np.zeros((image.height, image.width))
        min_dim = min(image.height, image.width)
        s[:min_dim, :min_dim] = np.diag(s_vector)
        return cls(image.width, image.height, u, s, v)



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
