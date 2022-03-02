"""
"""

import pandas as pd
import matplotlib.pyplot as plt
import os.path
import argparse
import numpy as np

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from sklearn.neighbors import KernelDensity
from PIL import Image
from kde_densities import parse_gaze_samples, element_label_densities, plot_density_overlay
from flipping_rate_analysis import flipping_candidate_score, flipping_candidate_score_of_rank


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

mycmap = transparent_cmap(plt.cm.Reds)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_file", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--show_density_overlay", action='store_true')
    args = vars(parser.parse_args())

    vis = args['fix_file'].split('/')[-3]

    element_labels = parse_element_label(os.path.join(args['element_labels_dir'], vis))
    element_labels = combine_rows(element_labels)
    img_path = os.path.join(args['images_dir'], vis + '.png')

    bandwidth = 1.
    plt.figure(figsize=(16, 16))

    with Image.open(img_path) as im:
        plt.imshow(im)
        width, height = im.size  # original image size
        X,Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.full((width, height), 0, dtype="int")

        fixations = pd.read_csv(args['fix_file'], header=None)
        for index, row in fixations.iterrows():
            # Gaze XY
            gaze_samples = parse_gaze_samples(row[4], row[5])
            # Step 4: Perform KDE on gaze samples associated to fixation
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)
            # Step 5: check which AOIs are overlaid by the resulting density and to which extent
            fixation_density = element_label_densities(kde, element_labels, im.size)
            # Filter out small densities
            fixation_density = list(filter(lambda x: x[1] > 1e-6, fixation_density))
            sum_densities = sum([d for _, d in fixation_density])
            if sum_densities < 1.:
                fixation_density.append(('#', 1.-sum_densities))

            score_r2 = flipping_candidate_score_of_rank(fixation_density, r=2)
            score_r3 = flipping_candidate_score_of_rank(fixation_density, r=3)
            score_r4 = flipping_candidate_score_of_rank(fixation_density, r=4)

            if score_r4 > 0.7 and score_r4 > score_r2 and score_r4 > score_r3:
                #fixation_density = list(filter(lambda x: x[1] > 1e-6, fixation_density))
                #print(fixation_density)
                #print(f'score_r2 = {score_r2}, score_r3={score_r3}, score_r4={score_r4}\n')
                z = np.exp(kde.score_samples(xy))
                Z = Z + z.reshape(Z.shape)
                plt.plot(row[1], row[2], 'bx')

        Z = Z.reshape(X.shape)
        levels = np.linspace(0, Z.max(), 500)
        plt.contourf(X, Y, Z, levels=levels, cmap=mycmap)
        plt.show()
