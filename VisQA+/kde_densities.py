"""
Steps 4 and 5: Computes to what extent the density associated to each fixations overlaps with the interior of neighboring AOIs.
The density is computed from KDE applied to gaze samples associated to the fixation.
The output, categorized by vis type, is stored in 'densityByVis/'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import argparse
import pandas as pd
import json

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from VisQA.dataset.dataset_io import convert_desc
from sklearn.neighbors import KernelDensity
from skimage.draw import polygon
from PIL import Image
from glob import glob
from tqdm import tqdm
from os import makedirs


def parse_gaze_samples(xp_str, yp_str):
    xps = xp_str[1:-1].split(';')
    yps = yp_str[1:-1].split(';')

    xps = np.array(xps).astype(float)
    yps = np.array(yps).astype(float)
    return np.stack((xps, yps), axis=1)


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


def element_label_densities(kde, element_labels, size):
    densities = []
    width, height = size

    for row in element_labels.iterrows():
        #row[1][0] id
        #row[1][1] desc
        #row[1][2] file
        #row[1][3] coordinates
        # Polygon expects Y,X
        rr, cc = polygon(
            np.array(row[1][3])[:,1],
            np.array(row[1][3])[:,0],
            (height, width))
        # KDE expects X,Y
        in_poly = np.vstack((cc, rr)).T
        acc_density = np.exp(kde.score_samples(in_poly)).sum()
        label = convert_desc(row[1][1])
        densities.append((label, acc_density))

    sum_densities = sum([d for _, d in densities])
    # TODO How should we handle the overlapping case and to which extent it becomes a problem?
    #assert sum_densities <= 1+1e-1, f"Element labels are overlapping. Densities add up to {sum_densities:.3f}"
    # Artificial label that covers the rest of the visual space
    return densities


def plot_density_overlay(kde, im):
    mycmap = transparent_cmap(plt.cm.Reds)
    plt.figure(figsize=(16, 16))

    width, height = im.size  # original image size

    X,Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    Z = np.full((width, height), 0, dtype="int")
    Z = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    if(Z.max()==0):
        return

    plt.imshow(im)
    levels = np.linspace(0, Z.max(), 25)
    cb = plt.contourf(X, Y, Z, levels=levels, cmap=mycmap)
    plt.colorbar(cb)


def calc_densities(im, fixations, element_labels, bandwidth=1.0, show_density_overlay=True):
    """
    Calculate overlap between given AOIs and densities computed from KDE step.
    Output can be verified with show_density_overlay=True
    """
    densities = {}
    for index, row in fixations.iterrows():
        # Gaze XY
        gaze_samples = parse_gaze_samples(row[4], row[5])
        # Step 4: Perform KDE on gaze samples associated to fixation
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)
        if show_density_overlay:
            plot_density_overlay(kde, im)
            plt.plot(row[1], row[2], 'bx')
            plt.show()
        # Step 5: check which AOIs are overlaid by the resulting density and to which extent
        fixation_density = element_label_densities(kde, element_labels, im.size)
        # Filter zero densities
        fixation_density = list(filter(lambda d: d[1] > 0.0, fixation_density))
        densities[index] = fixation_density
    return densities


def densities_of_vis(vis_path, out_dir, im, element_labels, show_density_overlay):
    vis = os.path.basename(vis_path)

    for fix_path in tqdm(glob(os.path.join(vis_path, 'enc', '*.csv')), desc=f'{vis}', unit='csv files'):
        fixations = pd.read_csv(fix_path)
        densities = calc_densities(im, fixations, element_labels, show_density_overlay=show_density_overlay)

        filename = os.path.join(out_dir, os.path.basename(fix_path)[:-4])
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(densities, f, ensure_ascii=False)


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--element_labels_dir", type=str)
    parser.add_argument("--show_density_overlay", action='store_true')
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    # Directory containing densities categorized by type/vis
    root_dir = os.path.join(args['dataset_dir'], 'densitiesByVis')
    makedirs(root_dir, exist_ok=True)

    # NOTE: Discard images that do not meet certain constraints
    ok_images = []
    with open('vis_ok', 'r') as f:
        ok_images.extend([line.replace('\n', '') for line in f.readlines()])

    for vis_type in vis_types:
        vis_type_dir = os.path.join(args['dataset_dir'], 'eyetracking', 'csv_files', 'fixationsByVis', vis_type)

        for vis_path in glob(os.path.join(vis_type_dir, '*')):
            vis = os.path.basename(vis_path)
            if vis not in ok_images:
                continue
            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
            element_labels = combine_rows(element_labels)
            img_path = os.path.join(args['images_dir'], vis + '.png')

            # Create vis type directory
            vis_dir = os.path.join(root_dir, vis_type, vis)
            makedirs(vis_dir, exist_ok=True)

            with Image.open(img_path) as im:
                densities_of_vis(vis_path, vis_dir, im, element_labels, args['show_density_overlay'])