import re, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer
from bs4 import BeautifulSoup
from libtiff import TIFF
from PIL import Image



def read_reuters21578(dir, vectorizer=TfidfVectorizer()):
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in sorted(files) if re.match(r"reut2.*\.sgm", f)]
    stories = []
    for f in files:
        print(f)
        text = open(f, "r", errors="ignore").read()
        soup = BeautifulSoup(text, "html.parser")
        for text_block in soup.find_all("text"):
            body = text_block.find("body")
            if body:
                stories.append(body.get_text())
            else:
                stories.append(text_block.get_text())
    X = vectorizer.fit_transform(stories)
    return X


def read_pines(dir):
    read_tiff_func = lambda path: TIFF.open(path).read_image()

    ns_line_path = os.path.join(dir, "19920612_AVIRIS_IndianPine_NS-line.tif")
    ns_line_gt_path = os.path.join(dir, "19920612_AVIRIS_IndianPine_NS-line_gr.tif")
    site3_path = os.path.join(dir, "19920612_AVIRIS_IndianPine_Site3.tif")
    site3_gt_path = os.path.join(dir, "19920612_AVIRIS_IndianPine_Site3_gr.tif")

    ns_line_im = read_tiff_func(ns_line_path)
    ns_line_gt_im = read_tiff_func(ns_line_gt_path)
    site3_im = read_tiff_func(site3_path)
    site3_gt_im = read_tiff_func(site3_gt_path)

    return dict(
        ns_line_im=ns_line_im,
        ns_line_gt_im=ns_line_gt_im,
        site3_im=site3_im,
        site3_gt_im=site3_gt_im
    )


def read_face_images(dir):
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in sorted(files)]
    faces = []
    for f in files:
        face = np.array(Image.open(f))
        faces.append(face)
    faces = np.array(faces)
    return faces


def unroll_images(data):
    original_shape = data.shape
    return data.reshape(original_shape[0], -1), (original_shape[1], original_shape[2])


def roll_images(data, original_image_shape):
    if data.ndim == 1:
        return data.reshape(*original_image_shape)
    else:
        return data.reshape(-1, *original_image_shape)


def images_matrix_grid(data, grid_shape):
    imrows = [
        np.hstack([data[i, :, :]
                   for i in range(grid_shape[1] * a, grid_shape[1] * a + grid_shape[1])])
        for a in range(0, grid_shape[0])
    ]
    return np.vstack(imrows)


class HashTfidfVectoriser:
    def __init__(self, n_features):
        self.hashing_vectoriser = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self.tfidf_transformer = TfidfTransformer()

    def fit_transform(self, data):
        return self.tfidf_transformer.fit_transform(self.hashing_vectoriser.fit_transform(data))
