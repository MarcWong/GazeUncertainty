"""
Create a list of visualizations whose AOIs have minimal overlap.
Additionally a maximum image size constraint can be provdided to filter out large images (that cause long computation time in 'kde_densities.py').
The output is stored in file '[VIS_TYPE]_included'.
"""

import argparse
import os.path

from glob import glob
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from aoi_overlap import get_overlap
from PIL import Image


def get_image_list(args):
    images = []

    for vis_path in glob(os.path.join(args['dataset_dir'], 'eyetracking', 'csv_files', 'fixationsByVis', '*', '*')):
        vis_type = os.path.basename(os.path.dirname(vis_path))
        if not vis_type in args['vis_type']:
            continue

        vis = os.path.basename(vis_path)
        img_path = os.path.join(args['images_dir'], vis + '.png')

        if not os.path.exists(img_path):
            img_path = os.path.join(args['images_dir'], vis + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(args['images_dir'], vis + '.jpeg')
        if not os.path.exists(img_path):
            print(f'Image of {vis} not found in {args["images_dir"]}')
            exit()
        
        element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
        element_labels = combine_rows(element_labels)

        with Image.open(img_path) as im:
            width, height = im.size
            overlap = get_overlap(width, height, element_labels)
            overlap_ratio = overlap.sum() / (width * height)
            size_constraint = width * height <= args['max_pixels']
            if size_constraint:
                images.append((vis, img_path, overlap_ratio))
    return images

if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--vis_type", choices=VIS_TYPES)
    parser.add_argument("--num_output", type=int, required=True)
    parser.add_argument("--max_pixels", type=int, default=500*400)
    args = vars(parser.parse_args())

    images = get_image_list(args)

    if len(images) == 0:
        print('No images found!')
        exit()

    # Sort criterion: Overlap between AOIs
    images.sort(key=lambda x: x[2], reverse=False)
    # Take the best images
    images = images[:args['num_output']]

    with open(f'included_{args["vis_type"]}', 'w') as f:
        for vis, img_path, overlap_ratio in images:
            f.write(f'{vis}\n')
            #f.write(f'{vis}, {overlap_ratio:.3f}\n')
