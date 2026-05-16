import os
import cv2
import json
import numpy as np

DATA_PATH = r"C:\Users\halam\Downloads\Image processing project\data"

img_dir = os.path.join(DATA_PATH, "images")
label_dir = os.path.join(DATA_PATH, "labels")

with open(os.path.join(DATA_PATH, "train_list.json")) as f:
    train_list = json.load(f)

with open(os.path.join(DATA_PATH, "test_list.json")) as f:
    test_list = json.load(f)


def load_label(img_name):
    label_file = img_name.replace(".png", ".txt")
    path = os.path.join(label_dir, label_file)
    return float(np.loadtxt(path))


def load_image(img_name):
    path = os.path.join(img_dir, img_name)
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128,128))
    return img