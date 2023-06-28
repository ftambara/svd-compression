"""
Code used to render color-scale matrices.
"""

import matplotlib.pyplot as plt

from image import RawImage


def display_image(image: RawImage) -> None:
    """
    Display the image in a new window.
    """
    plt.imshow(image.data, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()