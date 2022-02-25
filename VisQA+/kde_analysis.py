import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os.path

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from VisQA.dataset.dataset_io import convert_desc
from VisQA.preprocessing.match.match_eye_fixations_to_element_labels import compute_is_in_polygon
from sklearn.neighbors import KernelDensity
from skimage.draw import polygon


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


def element_label_densities(kde, element_labels, size, density_threshold=1e-5):
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

        if acc_density >= density_threshold:
            label = convert_desc(row[1][1])
            densities.append((label, acc_density))
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


if __name__ == '__main__':
    img_src = '_Di4yk2H64QEBkEncRGMhg==.0'
    #img_dir = '/netpool/homes/wangyo/Dataset/VisQA/merged/src/'
    img_dir = 'C:/Users/kochme/Datasets/VisQA/merged/src_bb/'
    img_path = os.path.join(img_dir, img_src + '.png')

    fixation_path = f'C:/Users/kochme/Datasets/VisQA/eyetracking/csv_files/fixationsByVis/{img_src}/enc/byq1.csv'
    #fixation_path = f'/netpool/homes/wangyo/Dataset/VisQA/eyetracking/csv_files/fixationsByVis/{img_src}/enc/byq1.csv'
    fixation = pd.read_csv(fixation_path)

    #element_labels = parse_element_label(f'/netpool/homes/wangyo/Dataset/VisQA/element_labels/{img_src}')
    element_labels = parse_element_label(f'C:/Users/kochme/Datasets/VisQA/element_labels/{img_src}')
    element_labels = combine_rows(element_labels)

    bandwidth = 1


    with Image.open(img_path) as im:
        plt.show()
        for index, row in fixation.iterrows():
            # Gaze XY
            gaze_samples = parse_gaze_samples(row[4], row[5])

            # Step 4: Perform KDE on gaze samples associated to fixation
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)
            # Step 5: check which AOIs are overlaid by the resulting density and to which extent
            densities = element_label_densities(kde, element_labels, im.size)
            print(densities)

            plot_density_overlay(kde, im)
            plt.plot(row[1], row[2], 'bx')
            plt.show()

