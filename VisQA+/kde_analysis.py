import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from VisQA.preprocessing.parser.parse_element_labels import parse_element_label, combine_rows
from VisQA.dataset.dataset_io import convert_desc
from VisQA.preprocessing.match.match_eye_fixations_to_element_labels import compute_is_in_polygon
from sklearn.neighbors import KernelDensity
from skimage.draw import polygon


def get_hit_aoi(element_labels, xp, yp):
    for _, row in element_labels.iterrows():
        inp = {'axp': xp, 'ayp': yp, 'polygon': row['polygon'], 'desc': row['desc']}
        if compute_is_in_polygon(inp)['contains']:
            return convert_desc(row['desc'])
    # Not within defined any AOI
    return '#'


def parse_gaze_samples(xp_str, yp_str):
    xps = xp_str[1:-1].split()
    yps = yp_str[1:-1].split()

    xps = np.array(xps).astype(int)
    yps = np.array(yps).astype(int)
    return np.stack((xps, yps), axis=1)


def check_consistency(fix, gaze_samples):
    avg_x, avg_y = np.round(np.mean(gaze_samples, axis=0)).astype(int)
    fix_x, fix_y = int(fix[0]), int(fix[1])
    #print(f'avg_x: {avg_x}; fix_x: {fix_x}; avg_y: {avg_y}; fix_y: {fix_y}')
    assert abs(avg_x - fix_x) <= 1 and abs(avg_y - fix_y) <= 1, "Consistency check failed!"


if __name__ == '__main__':
    img_src = '_Di4yk2H64QEBkEncRGMhg==.0'
    img_dir = '/netpool/homes/wangyo/Dataset/VisQA/merged/src/'

    #fixation_path = f'C:/Users/kochme/Datasets/VisQA/eyetracking/csv_files/fixationsByVis/{img_src}/enc/byq1.csv'
    fixation_path = f'/netpool/homes/wangyo/Dataset/VisQA/eyetracking/csv_files/fixationsByVis/{img_src}/enc/byq1.csv'
    fixation = pd.read_csv(fixation_path)

    element_labels = parse_element_label(f'/netpool/homes/wangyo/Dataset/VisQA/element_labels/{img_src}')
    element_labels = combine_rows(element_labels)
    #print(element_labels)

    bandwidth = 1

    for index, row in fixation.iterrows():
        # Gaze Left XY
        gaze_samples = parse_gaze_samples(row[4], row[5])

        # Step 4: Perform KDE on gaze samples associated to fixation
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(gaze_samples)

        # Step 5: check which AOIs are overlaid by the resulting density and to which extent
        with Image.open(img_dir+img_src+'.png') as im:
            #fig = plt.figure(figsize=(16, 16))
            #plt.imshow(im)
            width, height = im.size  # original image size

            X,Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
            xy = np.vstack([Y.ravel(), X.ravel()]).T

            Z = np.full((width, height), 0, dtype="int")
            Z = np.exp(kde.score_samples(xy))
            Z = Z.reshape(X.shape)

            for row in element_labels.iterrows():
                #row[1][0] id
                #row[1][1] desc
                #row[1][2] file
                #row[1][3] coordinates
                rr, cc = polygon(
                    np.array(row[1][3])[:,1],
                    np.array(row[1][3])[:,0],
                    (width, height))
                try:
                    print(rr.max(), cc.max())
                    print(Z[rr,cc].sum()) # this should be the accumulated density of all pixels within that polygon
                except: break

            #levels = np.linspace(0, Z.max(), 25)
            #plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
            #plt.show()