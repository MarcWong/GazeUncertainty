import os
import pandas as pd
import numpy as np

from tqdm import trange
from glob import glob
from natsort import natsorted


def scanpath_stat_enc():
    dataset_dir = "/Users/ruhdorfer/dev/VisQA+/DATASET/VisQA"
    fixationsByVis = f"{dataset_dir}/eyetracking/csv_files/fixationsByVis"
    curfiles = natsorted(glob(fixationsByVis + '/*'))
    excluded = pd.read_csv("dataset/excluded.csv")
    excluded_subjects = excluded['subject_id'].unique()

    y_total = []
    x_total = []
    d_total = []
    l_total = []
    viewers = []

    for id in trange(0, len(curfiles), 1):
        allfiles = glob(os.path.join(curfiles[id], 'enc', '*.csv'))

        viewers.append(len(allfiles))
        for csv in allfiles:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])

            x = fixations['x'].tolist()
            y = fixations['y'].tolist()
            duration = fixations['time'].tolist()
            x_total.extend(x)
            y_total.extend(y)
            d_total.extend(duration)
            l_total.append(len(x))

    y_total = np.array(y_total)
    x_total = np.array(x_total)
    d_total = np.array(d_total)
    l_total = np.array(l_total)
    viewers = np.array(viewers)

    print("Duration")
    print(np.mean(d_total))
    print(np.std(d_total))

    print("Length")
    print(np.mean(l_total))
    print(np.std(l_total))
    print(np.max(l_total))
    print(np.min(l_total))

    print("Viewers")
    print(np.mean(viewers))


def scanpath_stat_recall():
    dataset_dir = "/Users/ruhdorfer/dev/VisQA+/DATASET/VisQA"
    fixationsByVis = f"{dataset_dir}/eyetracking/csv_files/fixationsByVis"
    curfiles = natsorted(glob(fixationsByVis + '/*'))
    excluded = pd.read_csv("dataset/excluded.csv")
    excluded_subjects = excluded['subject_id'].unique()

    l_total_1 = []
    l_total_2 = []
    l_total_3 = []
    l_total_4 = []
    l_total_5 = []

    for id in trange(0, len(curfiles), 1):
        allfiles_enc = glob(os.path.join(curfiles[id], 'enc', '*.csv'))
        allfiles_1 = glob(os.path.join(curfiles[id], 'recall', '1', '*.csv'))
        allfiles_2 = glob(os.path.join(curfiles[id], 'recall', '2', '*.csv'))
        allfiles_3 = glob(os.path.join(curfiles[id], 'recall', '3', '*.csv'))
        allfiles_4 = glob(os.path.join(curfiles[id], 'recall', '4', '*.csv'))
        allfiles_5 = glob(os.path.join(curfiles[id], 'recall', '5', '*.csv'))

        for csv in allfiles_1:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])
            x = fixations['x'].tolist()
            l_total_1.append(len(x))
        if len(allfiles_1) != len(allfiles_enc):
            l_total_1.extend(
                [0 for _ in range(0, len(allfiles_enc)-len(allfiles_1))])

        for csv in allfiles_2:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])
            x = fixations['x'].tolist()
            l_total_2.append(len(x))
        if len(allfiles_2) != len(allfiles_enc):
            l_total_2.extend(
                [0 for _ in range(0, len(allfiles_enc)-len(allfiles_1))])

        for csv in allfiles_3:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])
            x = fixations['x'].tolist()
            l_total_3.append(len(x))
        if len(allfiles_3) != len(allfiles_enc):
            l_total_3.extend(
                [0 for _ in range(0, len(allfiles_enc)-len(allfiles_1))])

        for csv in allfiles_4:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])
            x = fixations['x'].tolist()
            l_total_4.append(len(x))
        if len(allfiles_4) != len(allfiles_enc):
            l_total_4.extend(
                [0 for _ in range(0, len(allfiles_enc)-len(allfiles_1))])

        for csv in allfiles_5:
            if os.path.basename(csv)[:-4] in excluded_subjects:
                continue
            fixations = pd.read_csv(csv, header=None, names=[
                'index', 'x', 'y', 'time'])
            x = fixations['x'].tolist()
            l_total_5.append(len(x))
        if len(allfiles_5) != len(allfiles_enc):
            l_total_5.extend(
                [0 for _ in range(0, len(allfiles_enc)-len(allfiles_1))])
    print(np.mean(l_total_1))
    print(np.max(l_total_1))
    print(np.mean(l_total_2))
    print(np.std(l_total_2))
    print(np.mean(l_total_3))
    print(np.std(l_total_3))
    print(np.mean(l_total_4))
    print(np.std(l_total_4))
    print(np.mean(l_total_5))
    print(np.std(l_total_5))


if __name__ == '__main__':
    scanpath_stat_recall()
