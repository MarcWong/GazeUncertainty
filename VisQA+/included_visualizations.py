"""
Filter out visualizations accoarding to the following constraints:

(1) Size constraint: maximum number of pixels in image
(2) Overlap constraint: maximum overlap among AOIs

Visualizations that meet those criteria are written to 'vis_ok' file.
"""

import argparse
import os.path

from glob import glob
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from aoi_overlap import get_overlap
from PIL import Image
import json


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')

    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--images_spec_dir", type=str, required=True)
    parser.add_argument("--vis_type", choices=VIS_TYPES)
    parser.add_argument("--max_pixels", type=int, default=500*400)
    parser.add_argument("--max_overlap_ratio", type=float, default=0.01)
    args = vars(parser.parse_args())

    type2file = {}

    for img_path in glob(os.path.join(args['images_dir'], '*.png')):
        vis = os.path.basename(img_path)[:-4]
        with open(os.path.join(args['images_spec_dir'], vis + '.json')) as spec:
            vis_type = json.load(spec)['vistype']
            if vis_type != args['vis_type']:
                continue

            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
            element_labels = combine_rows(element_labels)

            if vis_type not in type2file:
                type2file[vis_type] = open(f'included_{vis_type}', 'w')

            with Image.open(img_path) as im:
                width, height = im.size
                overlap = get_overlap(width, height, element_labels)
                overlap_ratio = overlap.sum() / (width * height)

                size_constraint = width * height <= args['max_pixels']
                overlap_constraint = overlap_ratio <= args['max_overlap_ratio']
                if size_constraint and overlap_constraint:
                    #type2file[vis_type].write(f'{vis}, {overlap_ratio:.09f}, {width*height}\n')
                    type2file[vis_type].write(f'{vis}\n')
 
    for file in type2file.values():
        file.close()
