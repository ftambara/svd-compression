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
    s_vector: np.ndarray
    v: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """
        Return the reconstructed image data.
        """
        s_height = self.u.shape[1]
        s_width = self.v.shape[0]
        s = np.zeros((s_height, s_width))

        # Fill in the diagonal of s
        for i, s_i in enumerate(self.s_vector):
            s[i, i] = s_i

        # Reconstruct the image
        return self.u @ s @ self.v


    def theoretical_compression_ratio(self) -> float:
        """
        Return the theoretical data compression ratio for this image.
        """
        full_size = self.width * self.height
        compressed_size = self.u.size + self.s_vector.size + self.v.size
        return full_size / compressed_size

    def keep_n_components(self, n: int) -> "SVDImage":
        """
        Return a new SVDImage with only the first n components.
        """
        if not 0 < n <= min(self.width, self.height):
            raise ValueError(
                f"n must be between 0 and {min(self.width, self.height)}, inclusive"
            )
        return SVDImage(
            self.width, self.height, self.u[:, :n], self.s_vector[:n], self.v[:n, :]
        )

    @classmethod
    def from_raw_image(cls, image: RawImage) -> "SVDImage":
        u, s_vector, v = la.svd(image.data)
        return cls(image.width, image.height, u, s_vector, v)


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
