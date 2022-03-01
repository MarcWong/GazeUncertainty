"""
Drops enclosing AOIs, i.e. AOIs that are guaranteed to enclose another AOI, such as 'data' or 'graphical element'.
The updated label definitions are written to 'element_labels_fixed/'.
"""

import argparse
import os.path
import pandas as pd
import numpy as np

from os import makedirs
from glob import glob
from skimage.draw import polygon2mask
from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows


def get_polygon_masks(element_labels):
    min_coord = element_labels.apply(lambda row: np.min(row['polygon'], axis=0), result_type='expand', axis=1)
    max_coord = element_labels.apply(lambda row: np.min(row['polygon'], axis=0), result_type='expand', axis=1)

    min_x, min_y = min_coord.min()
    max_x, max_y = max_coord.max()

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    masks = []

    for row in element_labels.iterrows():
        poly = np.array(row[1][3])
        if len(poly) > 1:
            mask = polygon2mask((height, width), poly[:, ::-1])
            masks.append(mask)
    return masks


def get_enclosing_aois(masks):
    N = len(masks)
    enclosing_aoi = []
    for i in range(N):
        for j in range(N):
            if i != j:
                intersection = (masks[i] & masks[j]).sum()
                if intersection > 0 and intersection == masks[i].sum():
                    enclosing_aoi.append(j)
                elif intersection > 0 and intersection == masks[j].sum():
                    enclosing_aoi.append(i)
    return set(enclosing_aoi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, default=None)
    args = vars(parser.parse_args())

    cleaned_dir = os.path.join(os.path.dirname(args['element_labels_dir']), 'element_labels_fixed')
    makedirs(cleaned_dir, exist_ok=True)

    for path in glob(os.path.join(args['element_labels_dir'], '*')):
        """
        element_labels = parse_element_label(path)
        element_labels = combine_rows(element_labels)

        enclosing_aoi = get_enclosing_aois(get_polygon_masks(element_labels))
        for idx in set(enclosing_aoi):
            print(f'\t{element_labels.loc[idx, "desc"]}')
        element_labels.to_csv(os.path.join(cleaned_dir, os.path.basename(path)), index=False)
        """

        element_labels = pd.read_csv(path)
        a = element_labels[element_labels.iloc[:, 1].isin(['data (area)', 'data', 'graphical element'])]
        element_labels.drop(a.index, inplace=True)
