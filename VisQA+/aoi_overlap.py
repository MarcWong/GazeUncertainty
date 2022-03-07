"""
Computes AOI overlap within each visualization. 
Images overlayed with bounding boxes and overlapping regions are written to 'src_bb/' directory.
"""
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs
from glob import glob
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from skimage.draw import polygon2mask
from PIL import Image, ImageDraw


def get_overlap(width, height, element_labels):
    canvas = np.zeros((height, width), dtype=np.bool8)
    overlap = np.zeros((height, width), dtype=np.bool8)

    for row in element_labels.iterrows():
        poly = np.array(row[1][3])
        if len(poly) > 1:
            mask = polygon2mask((height, width), poly[:, ::-1])
            overlap |= (canvas & mask)
            canvas |= mask
    return overlap


def draw_bounding_boxes(im, element_labels):
    im_draw = ImageDraw.Draw(im)
    for row in element_labels.iterrows():
        poly = np.array(row[1][3])
        if len(poly) > 1:
            im_draw.polygon(poly.flatten().tolist(), outline='blue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    args = vars(parser.parse_args())

    makedirs(os.path.join(os.path.dirname(args['images_dir']), 'src_bb'), exist_ok=True)

    for img_path in glob(os.path.join(args['images_dir'], '*.*')):
        vis = os.path.basename(img_path)[:-4]
        element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
        element_labels = combine_rows(element_labels)

        with Image.open(img_path) as im:
            width, height = im.size

            draw_bounding_boxes(im, element_labels)
            im = np.array(im)

            overlap = get_overlap(width, height, element_labels)
            overlap = overlap.astype(float) * .5

            im[:, :, 1] = im[:, :, 1] * (1 - overlap)
            im[:, :, 2] = im[:, :, 2] * (1 - overlap)

            plt.imsave(os.path.join(os.path.dirname(args['images_dir']), 'src_bb', os.path.basename(img_path)), im)
