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
import json
import logging

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from VisQA.dataset.dataset_io import convert_desc
from sklearn.neighbors import KernelDensity
from skimage.draw import polygon
from PIL import Image
from glob import glob
from tqdm import tqdm
from os import makedirs
from util import compute_scale_factor


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


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
    levels = np.linspace(0, Z.max(), 250)
    cb = plt.contourf(X, Y, Z, levels=levels, cmap=mycmap)
    plt.colorbar(cb)


def parse_gaze_samples(xp_str, yp_str):
    xps = xp_str[1:-1].split(';')
    yps = yp_str[1:-1].split(';')

    if len(xps) == 0 or len(yps) == 0:
        raise ValueError('Could not parse any gaze samples!')
    if xps == [''] or yps == ['']:
        raise ValueError('Could not parse any gaze samples!')

    xps = np.array(xps).astype(float)
    yps = np.array(yps).astype(float)
    return np.stack((xps, yps), axis=1)


def element_label_densities(kde, element_labels, size):
    densities = []
    width, height = size

    for id, row in element_labels.iterrows():
        #row[0] id
        #row[1] desc
        #row[2] file
        #row[3] coordinates
        # Polygon expects Y,X
        rr, cc = polygon(
            np.array(row[3])[:,1],
            np.array(row[3])[:,0],
            (height, width))
        # KDE expects X,Y
        in_poly = np.vstack((cc, rr)).T
        acc_density = np.exp(kde.score_samples(in_poly)).sum()
        label = convert_desc(row[1])
        densities.append((label, acc_density))
    return densities


def calc_densities(im, fixations, element_labels, bandwidth, show_density_overlay=True):
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

        sum_densities = sum([d for _, d in fixation_density])
        if sum_densities > 1.001:
            logging.warning(f'At fixation {index}: densities sum up to {sum_densities:.3f} > 1.0')

        densities[index] = fixation_density
    return densities


def densities_of_vis(vis_path, out_dir, im, element_labels, show_density_overlay):
    visname = os.path.basename(vis_path)
    img_path = os.path.join(args['images_dir'], visname + '.png')
    imgname = visname + '.png'
    if not os.path.exists(img_path):
        img_path = os.path.join(args['images_dir'], visname + '.jpg')
        imgname = visname + '.jpg'

    df_img_excluded = pd.read_csv(
        'dataset/excluded.csv')
    df_img_group = pd.read_csv(
        'dataset/image_annotation.csv')

    groupID = df_img_group[df_img_group['image'] == imgname].group.to_numpy()
    scale_factor = compute_scale_factor(im, groupID[0])
    #print(scale_factor)

    for fix_path in tqdm(glob(os.path.join(vis_path, 'enc', '*.csv')), desc=f'{visname}', unit='csv files'):
        try:
            subject_id = fix_path.split('/')[-1].strip('.csv')
            # 0.05 degree on screen
            BAND_WIDTH = 2.7
            if subject_id in df_img_excluded['subject_id'].values:
                # 0.25 degree on screen
                BAND_WIDTH *= 5
            BAND_WIDTH /= scale_factor
            fixations = pd.read_csv(fix_path, header=None)
            densities = calc_densities(im, fixations, element_labels, bandwidth=BAND_WIDTH, show_density_overlay=show_density_overlay)

            filename = os.path.join(out_dir, os.path.basename(fix_path)[:-4])
            with open(filename + '.json', 'w', encoding='utf-8') as f:
                json.dump(densities, f, ensure_ascii=False)
        except ValueError as e:
            logging.critical(f'Skip {fix_path}: {e}')


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--show_density_overlay", action='store_true')
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    #parser.add_argument("--bandwidth", type=float, required=True)
    parser.add_argument("--vis_included", type=str, default='vis_ok')
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    # Directory containing densities categorized by type/vis
    root_dir = os.path.join(args['dataset_dir'], 'densitiesByVis')
    makedirs(root_dir, exist_ok=True)

    # NOTE: Discard images that do not meet certain constraints
    ok_images = []
    with open(args['vis_included'], 'r') as f:
        ok_images.extend([line.replace('\n', '') for line in f.readlines()])

    logging_filename = f'kde_densities_{"-".join(vis_types)}.log'
    logging.basicConfig(filename=os.path.join(root_dir, logging_filename),
                        filemode='w',
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logging.info("".join(["*"] * 80))
    logging.info(f"dataset_dir: {args['dataset_dir']}")
    logging.info(f"images_dir: {args['images_dir']}")
    logging.info(f"element_labels_dir: {args['element_labels_dir']}")
    #logging.info(f"bandwidth: {args['bandwidth']:.2f}")
    logging.info(f"number of elements in vis_ok: {len(ok_images)}")
    logging.info("".join(["*"] * 80))

    for vis_type in vis_types:
        vis_type_dir = os.path.join(args['dataset_dir'], 'eyetracking', 'csv_files', 'fixationsByVis', vis_type)

        for vis_path in glob(os.path.join(vis_type_dir, '*')):
            vis = os.path.basename(vis_path)
            if vis not in ok_images:
                logging.info(f'Skip {vis} since it is not in vis_ok')
                continue

            element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
            element_labels = combine_rows(element_labels)

            img_path = os.path.join(args['images_dir'], vis + '.png')
            if not os.path.exists(img_path): img_path = os.path.join(args['images_dir'], vis + '.jpg')

            # Create vis type directory
            vis_dir = os.path.join(root_dir, vis_type, vis)
            makedirs(vis_dir, exist_ok=True)

            with Image.open(img_path) as im:
                #densities_of_vis(vis_path, vis_dir, im, element_labels, bandwidth=args['bandwidth'], show_density_overlay=args['show_density_overlay'])
                densities_of_vis(vis_path, vis_dir, im, element_labels, show_density_overlay=args['show_density_overlay'])
