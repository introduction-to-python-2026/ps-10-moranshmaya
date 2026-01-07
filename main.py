import os
import numpy as np
from PIL import Image
from image_utils import load_image, edge_detection


def save_image(arr, out_path):
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


def main():
    # בוחר תמונה קיימת בלי לשבור אוטוגריידר
    if os.path.exists("test.jpg"):
        image_path = "test.jpg"
    elif os.path.exists(".tests/lena.jpg"):
        image_path = ".tests/lena.jpg"
    else:
        print("No image found")
        return

    img = load_image(image_path)
    edges = edge_detection(img)

    save_image(edges, "edges.png")
    print("edges.png saved")


if _name_ == "_main_":
    main()
