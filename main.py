from PIL import Image
import numpy as np

from image_utils import load_image, edge_detection


def save_image(arr, out_path):
    # arr expected in [0,1]. Convert to uint8 [0,255] and save.
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(out_path)


def main():
    img = load_image("test.jpg")
    edges = edge_detection(img)
    save_image(edges, "edges.png")
    print("Saved edge image to edges.png")


if _name_ == "_main_":
    main()

