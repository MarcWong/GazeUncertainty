# -*- coding: utf-8 -*-
"""
Created on 20210320

@author: Yao Wang
@purpose: to compute the scanpath metrics of nc2020 predictions to the individual GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""
from scipy.spatial.distance import directed_hausdorff, euclidean
from fastdtw import fastdtw
import pandas as pd
import numpy as np
from glob import glob
import os
import sys
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
pd.options.mode.chained_assignment = None
sys.path.append("..")


def get_gt_files(imgname, extension, truncate=False):
    if truncate:
        gtdir = '../../eyetracking/csv_files/fixationsByVis_5000/%s/enc/' % imgname
    else:
        gtdir = '../../eyetracking/csv_files/fixationsByVis/%s/enc/' % imgname
    for path, subdir, files in os.walk(gtdir, extension):
        gt_files = files
    return path, gt_files


def DTW(P, Q, **kwargs):
    dist, _ = fastdtw(P, Q, dist=euclidean)
    return dist


def euclidean_distance(P, Q, **kwargs):
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
        P = P.astype(np.float32)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float32)
    elif Q.dtype != np.float32:
        Q = Q.astype(np.float32)
    if P.shape == Q.shape:
        return np.sqrt(np.sum((P-Q)**2))
    elif P.shape[1] == Q.shape[1]:
        min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
        return np.sqrt(np.sum((P[:min_len, :]-Q[:min_len, :])**2))

    return False


def TDE(
    P,
    Q,

    # options
    k=3,  # time-embedding vector dimension
    distance_mode='Mean', **kwargs
):
    """
            code reference:
                    https://github.com/dariozanca/FixaTons/
                    https://arxiv.org/abs/1802.02534

            metric: Simulating Human Saccadic Scanpaths on Natural Images.
                             wei wang etal.
    """

    # P and Q can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(P) < k or len(Q) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])

    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s_k_vec in Q_vectors:

        # find human k-vec of minimum distance

        norms = []

        for h_k_vec in P_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False


def scaled_TDE(
        H,
        S,
        image,

        # options
        toPlot=False):
    # to preserve data, we work on copies of the lists
    H_scanpath = np.copy(H)
    S_scanpath = np.copy(S)

    # First, coordinates are rescaled as to an image with maximum dimension 1
    # This is because, clearly, smaller images would produce smaller distances
    max_dim = float(max(np.shape(image)))

    for P in H_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    for P in S_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    # Then, scanpath similarity is computer for all possible k
    max_k = min(len(H_scanpath), len(S_scanpath))
    similarities = []
    for k in np.arange(1, max_k + 1):
        s = TDE(
            H_scanpath,
            S_scanpath,
            k=k,  # time-embedding vector dimension
            distance_mode='Mean')
        similarities.append(np.exp(-s))
        # print(similarities[-1])

    # Now that we have similarity measure for all possible k
    # we compute and return the mean

    if len(similarities) == 0:
        return 0
    return sum(similarities) / len(similarities)


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


fixationsByVis = "/Users/ruhdorfer/dev/VisQA+/DATASET/VisQA/eyetracking/csv_files/fixationsByVis"
strings = "/Users/ruhdorfer/dev/VisQA+/DATASET/VisQA/eyetracking/csv_files/strings/"
images_dir = "/Users/ruhdorfer/dev/VisQA+/DATASET/VisQA/merged/src"

excluded = pd.read_csv("dataset/excluded.csv")
excluded_subjects = excluded['subject_id'].unique()

ED_ENC_RECA = []
ED_ENC_RECO = []
ED_RECA_RECO = []

ST_ENC_RECA = []
ST_ENC_RECO = []
ST_RECA_RECO = []

for fixation in tqdm(os.listdir(strings)):
    if fixation in excluded_subjects:
        continue
    allfiles_enc = natsorted(
        glob(os.path.join(strings, fixation, 'enc', '*.txt')))
    subject_csvs = [os.path.basename(path) for path in allfiles_enc]
    allfiles_reco = natsorted(
        glob(os.path.join(
            strings, fixation, 'recognition', '*.txt')))
    allfiles_reco = [
        path for path in allfiles_reco if os.path.basename(path)
        in subject_csvs]
    allfiles_reca = natsorted(
        glob(os.path.join(
            strings, fixation, 'recall', '1', '*.txt')))

    for enc, reco, reca in zip(allfiles_enc, allfiles_reco, allfiles_reca):

        f = open(enc, 'r')
        scanpath_enc = f.readline()
        f = open(reco, 'r')
        scanpath_reco = f.readline()
        f = open(reca, 'r')
        scanpath_reca = f.readline()

        ST_ENC_RECA.append(nw_matching(scanpath_enc, scanpath_reca))
        ST_ENC_RECO.append(nw_matching(scanpath_enc, scanpath_reco))
        ST_RECA_RECO.append(nw_matching(scanpath_reca, scanpath_reco))


for fixation in tqdm(os.listdir(fixationsByVis)):
    if os.path.basename(fixation) == ".DS_Store":
        continue
    basepath = os.path.basename(fixation)
    imname, _ = os.path.splitext(basepath)

    png = os.path.join(images_dir, fixation+'.png')
    jpg = os.path.join(images_dir, fixation+'.jpg')

    if os.path.isfile(png):
        path = png
    elif os.path.isfile(jpg):
        path = jpg

    with Image.open(path) as im:
        width, height = im.size

        allfiles_enc = natsorted(
            glob(os.path.join(fixationsByVis, fixation, 'enc', '*.csv')))
        subject_csvs = [os.path.basename(path) for path in allfiles_enc]
        allfiles_reco = natsorted(
            glob(os.path.join(
                fixationsByVis, fixation, 'recognition', '*.csv')))
        allfiles_reco = [
            path for path in allfiles_reco if os.path.basename(path)
            in subject_csvs]
        allfiles_reca = natsorted(
            glob(os.path.join(
                fixationsByVis, fixation, 'recall', '1', '*.csv')))

        for enc, reco, reca in zip(allfiles_enc, allfiles_reco, allfiles_reca):
            if os.path.basename(enc)[:-4] in excluded_subjects:
                continue
            df_enc = pd.read_csv(enc, header=None)
            df_reco = pd.read_csv(reco, header=None)
            df_reca = pd.read_csv(reca, header=None)

            df_enc.columns = ['index', 'x', 'y', 'time']
            df_reco.columns = ['index', 'x', 'y', 'time']
            df_reca.columns = ['index', 'x', 'y', 'time']

            df_enc = df_enc.drop(['index'], axis=1)
            df_reco = df_reco.drop(['index'], axis=1)
            df_reca = df_reca.drop(['index'], axis=1)

            ED_ENC_RECA.append(DTW(df_enc.to_numpy(), df_reca.to_numpy()))
            ED_ENC_RECO.append(DTW(df_enc.to_numpy(), df_reco.to_numpy()))
            ED_RECA_RECO.append(DTW(df_reca.to_numpy(), df_reco.to_numpy()))

            # ED_ENC_RECA.append(euclidean_distance(
            #     df_enc.to_numpy(), df_reca.to_numpy()))
            # ED_ENC_RECO.append(euclidean_distance(
            #     df_enc.to_numpy(), df_reco.to_numpy()))
            # ED_RECA_RECO.append(euclidean_distance(
            #     df_reca.to_numpy(), df_reco.to_numpy()))

            # ED_ENC_RECA.append(scaled_TDE(
            #     df_enc.to_numpy(), df_reca.to_numpy(), im))
            # ED_ENC_RECO.append(scaled_TDE(
            #     df_enc.to_numpy(), df_reco.to_numpy(), im))
            # ED_RECA_RECO.append(scaled_TDE(
            #     df_reca.to_numpy(), df_reco.to_numpy(), im))

            # ED_ENC_RECA.append(TDE(df_enc.to_numpy(), df_reca.to_numpy()))
            # ED_ENC_RECO.append(TDE(df_enc.to_numpy(), df_reco.to_numpy()))
            # ED_RECA_RECO.append(TDE(df_reca.to_numpy(), df_reco.to_numpy()))

print("ST")
print("ENC - RECALL")
ST_ENC_RECA = np.asarray(ST_ENC_RECA, dtype=float)
print(np.round(np.mean(ST_ENC_RECA), 3))

print("ENC - RECOGNITION")
ST_ENC_RECO = np.asarray(ST_ENC_RECO, dtype=float)
print(np.round(np.mean(ST_ENC_RECO), 3))

print("RECALL - RECOGNITION")
ST_RECA_RECO = np.asarray(ST_RECA_RECO, dtype=float)
print(np.round(np.mean(ST_RECA_RECO), 3))

print("ED")

print("ENC - RECALL")
ED_ENC_RECA = np.asarray(ED_ENC_RECA, dtype=float)
print(np.round(np.mean(ED_ENC_RECA), 3))

print("ENC - RECOGNITION")
ED_ENC_RECO = np.asarray(ED_ENC_RECO, dtype=float)
print(np.round(np.mean(ED_ENC_RECO), 3))

print("RECALL - RECOGNITION")
ED_RECA_RECO = np.asarray(ED_RECA_RECO, dtype=float)
print(np.round(np.mean(ED_RECA_RECO), 3))
