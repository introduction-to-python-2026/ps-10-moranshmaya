import os
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


def save_image(arr, out_path):
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


def main():
    # choose an existing image if available
    if os.path.exists(".tests/lena.jpg"):
        in_path = ".tests/lena.jpg"
    elif os.path.exists("test.jpg"):
        in_path = "test.jpg"
    else:
        return  # don't crash if no image exists

    img = load_image(in_path)
    img = median(img, ball(3))
    edges = edge_detection(img)
    edges = (edges > 50).astype(np.uint8) * 255
    save_image(edges, "edges.png")
    print("Saved edge image to edges.png")


if __name__ == "__main__":
    main()
