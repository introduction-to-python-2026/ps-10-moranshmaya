from PIL import Image
import numpy as np

from image_utils import load_image, edge_detection


def save_image(arr, out_path):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(out_path)


def main():
    img = load_image("test.jpg")
    edges = edge_detection(img)
    save_image(edges, "edges.png")
    print("Saved edge image to edges.png")


if __name__ == "__main__":
    main()
