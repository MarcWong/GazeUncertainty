import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import os
import argparse
import pandas as pd

from sklearn.neighbors import KernelDensity
from PIL import Image
from glob import glob
from tqdm import tqdm
from util import transparent_cmap


def plot_density_overlay(kde, im):
    mycmap = transparent_cmap(plt.cm.Reds)
    width, height = im.size  # original image size

    X,Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    Z = np.full((width, height), 0, dtype="int")
    Z = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    if(Z.max()==0):
        return

    plt.imshow(im)
    levels = np.linspace(0, Z.max(), 50)
    plt.contourf(X, Y, Z, levels=levels, cmap=mycmap)


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


def fix_of_vis(vis_path):
    pos = []
    for fix_path in glob(os.path.join(vis_path, 'enc', '*.csv')):
        try:
            fixations = pd.read_csv(fix_path, header=None)
            pos.extend(zip(fixations[1].values, fixations[2].values))
        except ValueError as e:
           print(f'Skip {fix_path}: {e}')
    return np.array(pos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    args = vars(parser.parse_args())

    os.makedirs(os.path.join(os.path.dirname(args['images_dir']), 'src_heatmap'), exist_ok=True)

    for vis_path in tqdm(glob(os.path.join(args['dataset_dir'], 'eyetracking', 'csv_files', 'fixationsByVis', '*', '*')), unit='vis'):
        vis_type = os.path.basename(os.path.dirname(vis_path))
        vis = os.path.basename(vis_path)

        type_dir = os.path.join(os.path.dirname(args['images_dir']), 'src_heatmap', vis_type)
        os.makedirs(type_dir, exist_ok=True)

        img_path = os.path.join(args['images_dir'], vis + '.png')

        if not os.path.exists(img_path):
            img_path = os.path.join(args['images_dir'], vis + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(args['images_dir'], vis + '.jpeg')
        if not os.path.exists(img_path):
            print(f'Image of {vis} not found in {args["images_dir"]}')
            exit()

        with Image.open(img_path) as im:
            fixations = fix_of_vis(vis_path)
            kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(fixations)
            plot_density_overlay(kde, im)
            plt.tight_layout()
            plt.savefig(os.path.join(type_dir, os.path.basename(img_path)))
            plt.clf()