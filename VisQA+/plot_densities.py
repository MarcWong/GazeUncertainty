"""
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import argparse
import numpy as np

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from sklearn.neighbors import KernelDensity
from PIL import Image
from kde_densities import parse_gaze_samples, element_label_densities
from flipping_rate_analysis import flipping_candidate_score_of_rank
from util import compute_scale_factor, transparent_cmap

from tqdm import tqdm
from numpy.random import multivariate_normal


mycmap = transparent_cmap(plt.cm.Reds)



def kde_plot():
    mean = (5, 5)
    cov = np.ones((2, 2)) * 0.2
    cov[0, 0] = cov[1, 1] = 0.25

    gaze_samples = multivariate_normal(mean, cov, size=150)

    def plot_kde(ax, bandwidth):
        width, height = 10, 10
        X,Y = np.meshgrid(np.arange(0, width, 0.1), np.arange(0, height, 0.1))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.zeros_like(X)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)

        z = np.exp(kde.score_samples(xy))
        Z = Z + z.reshape(Z.shape)
        Z = Z.reshape(X.shape)

        levels = np.linspace(0, Z.max(), 300)
        ax.contourf(X, Y, Z, levels=levels, cmap=mycmap)
        
        for gx, gy in gaze_samples:
            ax.plot(gx, gy, 'b.', markerfacecolor=(1, 0, 0, 0.3))


    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 10)
    for gx, gy in gaze_samples:
        ax[0].plot(gx, gy, 'b.', markerfacecolor=(1, 0, 0, 1))

    plot_kde(ax[1], bandwidth=0.3)
    plot_kde(ax[2], bandwidth=0.7)
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_file", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--element_labels_dir", type=str, required=True)
    parser.add_argument("--target_ranks", type=int, default=(2, 3, 4), nargs='+')
    parser.add_argument("--threshold", type=float, default=0.5)
    args = vars(parser.parse_args())

    target_ranks = set(args['target_ranks'])
    vis = os.path.normpath(args['fix_file']).split(os.sep)[-3]

    element_labels = parse_element_label(os.path.join(os.path.normpath(args['element_labels_dir']), vis))
    element_labels = combine_rows(element_labels)
    img_path = os.path.join(args['images_dir'], vis + '.png')

    #kde_plot()
    #exit()
    plt.figure(figsize=(16, 16))

    df_img_excluded = pd.read_csv(
        'dataset/excluded.csv')
    df_img_group = pd.read_csv(
        'dataset/image_annotation.csv')

    subject_id = os.path.basename(args['fix_file']).strip('.csv')

    with Image.open(img_path) as im:
        groupID = df_img_group[df_img_group['image'] == vis + '.png'].group.to_numpy()
        scale_factor = compute_scale_factor(im, groupID[0])

        bandwidth = 2.7
        if subject_id in df_img_excluded['subject_id'].values:
            # 0.25 degree on screen
            bandwidth *= 5

        bandwidth /= scale_factor
        print(bandwidth)

        plt.imshow(im)
        width, height = im.size  # original image size
        X,Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.full((width, height), 0, dtype="int")

        fixations = pd.read_csv(args['fix_file'], header=None)
        for index, row in tqdm(list(fixations.iterrows()), unit='fixation'):
            # Gaze XY
            gaze_samples = parse_gaze_samples(row[4], row[5])
            # Step 4: Perform KDE on gaze samples associated to fixation
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)
            # Step 5: check which AOIs are overlaid by the resulting density and to which extent
            fixation_density = element_label_densities(kde, element_labels, im.size)

            rank_scores = {2: flipping_candidate_score_of_rank(fixation_density, r=2),
                           3: flipping_candidate_score_of_rank(fixation_density, r=3),
                           4: flipping_candidate_score_of_rank(fixation_density, r=4)}
            rank_of_max = max(rank_scores, key=rank_scores.get)

            fixation_density = list(filter(lambda x: x[1] > 1e-4, fixation_density))
            fixation_density = sorted(fixation_density, reverse=True, key=lambda x: x[1])
            #print(fixation_density)
            #print(rank_scores)
            #print('\n')

            if rank_scores[rank_of_max] > args['threshold'] and rank_of_max in target_ranks:
                print(fixation_density)
                print(rank_scores)
                z = np.exp(kde.score_samples(xy))
                Z = Z + z.reshape(Z.shape)
            plt.plot(row[1], row[2], 'rX')

        if Z.sum() > 0:
            Z = Z.reshape(X.shape)
            levels = np.linspace(0, Z.max(), 50)
            plt.contourf(X, Y, Z, levels=levels, cmap=mycmap)
            plt.show()
        else:
            print('Nothing to plot!')

        plt.show()