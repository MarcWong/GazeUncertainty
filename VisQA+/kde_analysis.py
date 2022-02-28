import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import argparse

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from VisQA.dataset.dataset_io import convert_desc
from sklearn.neighbors import KernelDensity
from skimage.draw import polygon
from PIL import Image
from glob import glob


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
        rr, cc = polygon(
            np.array(row[1][3])[:,1],
            np.array(row[1][3])[:,0],
            (height, width))
        in_poly = np.vstack((cc, rr)).T
        acc_density = np.exp(kde.score_samples(in_poly)).sum()
        label = convert_desc(row[1][1])
        densities.append((label, acc_density))

    sum_densities = sum([d for _, d in densities]) 
    assert sum_densities <= 1, f"Element labels are overlapping. Densities add up to {sum_densities:.3f}"
    # Artificial label that covers the rest of the visual space
    if sum_densities < 1.:
        densities.append(('#', 1.-sum_densities))
    return densities


def plot_density_overlay(kde, im):
    mycmap = transparent_cmap(plt.cm.Reds)
    fig = plt.figure(figsize=(16, 16))

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


def flipping_candidate_score(densities):
    """
    Flipping candidates score has the following interpretation:
    ~> 0: The density distribution is peaked, i.e. the fixation mostly covers just a single AOI.
    ~> 1: The density distribution is close to uniform, i.e. the fixation covers at least two AOI to a very similar extent.
    """
    def deviation_from_uniform(densities, N):
        uniform = 1 / N
        return sum([abs(d - uniform) for _, d in densities])

    sorted_densities = sorted(densities, reverse=True, key=lambda x: x[1])
    N = len(densities)
    max_score = 0

    # Flipping candidates switch between at least two AOIs
    for n in range(2, N+1):
        deviation = deviation_from_uniform(sorted_densities[:n], n)
        score = 1. - ((2 / n) * deviation)
        max_score = max(score, max_score)
    return max_score


def process(im, fixation, element_labels, show_density_overlay=True):
    """
    Perform KDE analysis steps. Calculated densities are used to find flipping candidates.
    Output can be verified with show_density_overlay=True
    """
    bandwidth = 1
    density_threshold = 0.01
    flipping_threshold = 0.5
    flipping_candidates = []

    for index, row in fixation.iterrows():
        # Gaze XY
        gaze_samples = parse_gaze_samples(row[4], row[5])
        # Step 4: Perform KDE on gaze samples associated to fixation
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)
        # Step 5: check which AOIs are overlaid by the resulting density and to which extent
        densities = element_label_densities(kde, element_labels, im.size)
        # Filter out small densities
        densities = list(filter(lambda x: x[1] >= density_threshold, densities))
        # Step 6: check for which segments the distribution overlays at least two AOIs to a very similar extent (the flipping candidates)
        if len(densities) > 1:
            score = flipping_candidate_score(densities)
            print(f'Flipping candidate score = {score:.3f}: {densities}')
            if score >= flipping_threshold:
                flipping_candidates.append((densities))
                if show_density_overlay:
                    plot_density_overlay(kde, im)
                    plt.plot(row[1], row[2], 'bx')
                    plt.show()
    return flipping_candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--vis_type", type=str, default='scatter')
    parser.add_argument("--show_density_overlay", action='store_true')
    args = vars(parser.parse_args())

    ok_images = []
    with open('no_overlap_vis', 'r') as f:
        ok_images.extend([line.replace('\n', '') for line in f.readlines()])

    #img_dir = '/netpool/homes/wangyo/Dataset/VisQA/merged/src/'
    #img_path = os.path.join(args['images_dir'], args['vis'].split('/')[1] + '.png')

    #fixation_path = f'{args["dataset_dir"]}/eyetracking/csv_files/fixationsByVis/{args["vis"]}/enc/{args["fixation"]}.csv'
    #fixation_path = f'/netpool/homes/wangyo/Dataset/VisQA/eyetracking/csv_files/fixationsByVis/{img_src}/enc/hbw1.csv'
    #fixation = pd.read_csv(fixation_path)

    #element_labels = parse_element_label(f'/netpool/homes/wangyo/Dataset/VisQA/element_labels/{img_src}')
    #element_labels = parse_element_label(f'{args["dataset_dir"]}/element_labels/{args["vis"].split("/")[1]}')
    #element_labels = combine_rows(element_labels)

    for vis_path in glob(os.path.join(args['dataset_dir'], 'eyetracking', 'csv_files', 'fixationsByVis', args['vis_type'], '*')):
        vis = os.path.basename(vis_path)
        img_path = os.path.join(args['images_dir'], vis + '.png')

        if vis not in ok_images:
            print(f'INFO: Skip vis {img_path}')
            continue

        element_labels = parse_element_label(os.path.join(args['dataset_dir'], 'element_labels', vis))
        element_labels = combine_rows(element_labels)

        with Image.open(img_path) as im:
            for fix_path in glob(os.path.join(vis_path, 'enc', '*.csv')):
                fixation = pd.read_csv(fix_path)
                print(f'********** Processing fixations \'{os.path.basename(fix_path)}\' on visualization \'{vis}\' **********\n')
                # For high-res visualizations this might be take some time!
                flipping_candidates = process(im, fixation, element_labels, show_density_overlay=args['show_density_overlay'])
                print(f'\n------> Number of flipping candidates: {len(flipping_candidates)}\n')