"""
Enlarges AOIs, i.e. make all AOIs in a visualisation bigger with radius param.
The updated label definitions are written to 'element_labels_enlarged/'.
"""

import argparse
import os.path
from tokenize import group
import pandas as pd
import numpy as np
from PIL import Image

from os import makedirs
from glob import glob

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from util import compute_scale_factor

# This is the real_on_screen px!

#0.05
#ENLARGE_PX = 2.7
#0.1
#ENLARGE_PX = 5.4
#0.15
#ENLARGE_PX = 8.1
#0.2
#ENLARGE_PX = 10.8
#0.25
#ENLARGE_PX = 13.5
#0.3
#ENLARGE_PX = 16.2
#0.4
#ENLARGE_PX = 21.6
#0.5
#ENLARGE_PX = 27
#0.6
#ENLARGE_PX = 32.4
#0.7
#ENLARGE_PX = 37.8
#0.8
#ENLARGE_PX = 43.2
#0.9
#ENLARGE_PX = 48.6
#1 degree
ENLARGE_PX = 54


# shape of polygon: [N, 2]
def Perimeter(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    permeter = 0.
    for i in range(N):
        permeter += np.linalg.norm(polygon[i-1] - polygon[i])
    return permeter

# Area of polygon
def Area(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    area = 0.
    vector_1 = polygon[1] - polygon[0]
    for i in range(2, N):
        vector_2 = polygon[i] - polygon[0]
        area += np.abs(np.cross(vector_1, vector_2))
        vector_1 = vector_2
    return area / 2

# |r| < 1
# r > 0, shrinking
# r < 0, enlarging
def calc_shrink_width(polygon: np.array, r):
    area = Area(polygon)
    perimeter = Perimeter(polygon)
    L = area * (1 - r ** 2) / perimeter

    return L if r > 0 else -L

def shrink_polygon(polygon: np.array, L, imgsize, fixWidth=False):
    N, d = polygon.shape
    width, height = imgsize
    if N < 3 or d != 2:
        raise ValueError

    shrinked_polygon = []

    if not fixWidth:
        L = calc_shrink_width(polygon, L)
    
    for i in range(N):
        Pi = polygon[i]
        v1 = polygon[i-1] - Pi
        v2 = polygon[(i+1)%N] - Pi

        normalize_v1 = v1 / np.linalg.norm(v1)
        normalize_v2 = v2 / np.linalg.norm(v2)

        sin_theta = np.abs(np.cross(normalize_v1, normalize_v2))

        Qi = Pi + L / sin_theta * (normalize_v1 + normalize_v2)
        Qi = np.maximum(Qi, 0)
        Qi[0] = np.where(Qi[0] < width - 1, Qi[0], width - 1)
        Qi[1] = np.where(Qi[1] < height - 1, Qi[1], height - 1)
        shrinked_polygon.append(np.array(Qi, dtype=int))
    return np.asarray(shrinked_polygon)

def enlargeAOI(element_labels, groupID, im, enlargeL:float):
    scale_factor = compute_scale_factor(im, groupID)
    realEnlargeL = enlargeL / scale_factor
    #print(im.size, realEnlargeL)

    df = pd.DataFrame(columns=['id', 'desc', 'x', 'y'])

    for id, row in element_labels.iterrows():
        #row[0] id
        #row[1] desc
        #row[2] file
        #row[3] coordinates
        if np.shape(np.array(row[3]))[0] < 3:
            for i in range(np.shape(np.array(row[3]))[0]):
                df = df.append({'id': row[0], 'desc': row[1], 'x': row[3][i][0], 'y': row[3][i][1]}, ignore_index=True)
            continue

        poly = np.array(row[3])
        #perimeter = Perimeter(poly)
        #area = Area(poly)

        expansion_poly = shrink_polygon(poly, realEnlargeL, im.size, fixWidth=True)
        #print(perimeter, area, expansion_poly)

        for i in range(np.shape(expansion_poly)[0]):
            df = df.append({'id': row[1][0], 'desc': row[1][1], 'x': expansion_poly[i][0], 'y': expansion_poly[i][1]}, ignore_index=True)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    args = vars(parser.parse_args())
    

    df_img_group = pd.read_csv(
        'dataset/image_annotation.csv')


    cleaned_dir = os.path.join(os.path.dirname(args['element_labels_dir']), f'element_labels_enlarged_{ENLARGE_PX}')
    makedirs(cleaned_dir, exist_ok=True)

    for path in glob(os.path.join(args['element_labels_dir'], '*')):
        visname = os.path.basename(path)
        img_path = os.path.join(args['images_dir'], visname + '.png')
        imgname = visname + '.png'
        if not os.path.exists(img_path):
            img_path = os.path.join(args['images_dir'], visname + '.jpg')
            imgname = visname + '.jpg'

        groupID = df_img_group[df_img_group['image'] == imgname].group.to_numpy()

        with Image.open(img_path) as im:
            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], visname))
            element_labels = combine_rows(element_labels)
            element_labels = enlargeAOI(element_labels, groupID[0], im, -ENLARGE_PX)
            element_labels.to_csv(os.path.join(cleaned_dir, os.path.basename(path)), header=False, index=False)
