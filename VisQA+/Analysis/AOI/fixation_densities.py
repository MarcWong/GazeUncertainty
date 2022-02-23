# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:51:58 2021

@author: Sruthi

Adpated by Constantin Ruhdorfer
"""
import os
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from tqdm import trange
from PIL import Image
from glob import glob
from skimage.draw import polygon
from natsort import natsorted
import seaborn as sns
import matplotlib.patheffects as pe

from visualization.plot_scanpaths import make_fixmap_and_heatmap
pd.options.mode.chained_assignment = None

matplotlib.use('agg')


BUCKETCOLORS = [
    (44/255., 105/255., 154/255., 0.5),
    (13/255., 179/255., 158/255., 0.5),
    (239/255., 234/255.,  90/255., 0.5),
    (241/255., 196/255.,  83/255., 0.5),
    (22/255., 219/255., 147/255., 0.5),
    (22/255., 219/255., 147/255., 0.5),
    (242/255., 158/255.,  76/255., 0.5),
    (44/255., 105/255., 154/255., 0.5),
    (4/255., 139/255., 168/255., 0.5),
    (131/255., 227/255., 119/255., 0.5),
    (84/255.,  71/255., 140/255., 0.5),
    (241/255., 196/255.,  83/255., 0.5),
    (185/255., 231/255., 105/255., 0.5),
    (185/255., 231/255., 105/255., 0.5),
    (128/255., 128/255., 128/255., 0.5)
]


class BBox:
    name = ''
    coords = None
    id = -1
    durations = None

    def __init__(self, name, coords, id):
        self.name = name
        self.coords = coords
        self.id = id


def get_accum_bucket(accum_time):
    accum_time = int(accum_time)
    if accum_time < 500:
        id2 = 0
    elif accum_time < 1000:
        id2 = 1
    elif accum_time < 1500:
        id2 = 2
    elif accum_time < 2000:
        id2 = 3
    elif accum_time < 2500:
        id2 = 4
    elif accum_time < 3000:
        id2 = 5
    elif accum_time < 3500:
        id2 = 6
    elif accum_time < 4000:
        id2 = 7
    elif accum_time < 4500:
        id2 = 8
    elif accum_time < 5000:
        id2 = 9
    elif accum_time < 5500:
        id2 = 10
    elif accum_time < 6000:
        id2 = 11
    elif accum_time < 6500:
        id2 = 12
    elif accum_time < 7000:
        id2 = 13
    elif accum_time < 7500:
        id2 = 14
    elif accum_time < 8000:
        id2 = 15
    elif accum_time < 8500:
        id2 = 16
    elif accum_time < 9000:
        id2 = 17
    elif accum_time < 9500:
        id2 = 18
    else:
        id2 = 19
    return id2


def get_bboxid_by_name(name):
    id1 = -1
    if 'annotation' in name:
        id1 = 0
    elif 'axis' in name:
        id1 = 1
    elif 'graphical element' in name:
        id1 = 2
    elif 'legend' in name:
        id1 = 3
    elif 'object' in name:
        if 'photograph' in name:
            id1 = 4
        else:  # 'pictogram'
            id1 = 5
    elif 'text' in name:
        if '(title)' in name:
            id1 = 6
        elif '(header row)' in name:
            id1 = 7
        elif '(label)' in name:
            id1 = 8
        elif '(paragraph)' in name:
            id1 = 9
        else:
            id1 = 10
    elif 'data (' in name:
        id1 = 11
    elif name == 'data':
        id1 = 12
    else:
        id1 = 13
    return id1


def get_bboxid_by_name_9(name):
    id1 = -1
    if 'annotation' in name:
        id1 = 0
    elif 'axis' in name:
        id1 = 1
    elif 'graphical element' in name:
        id1 = 2
    elif 'legend' in name:
        id1 = 3
    elif 'object' in name:
        id1 = 4
    elif 'text' in name:
        if ('(title)' in name
                or '(header row)' in name
                or '(paragraph)' in name):
            id1 = 5
        else:
            id1 = 6
    elif 'data' in name:
        id1 = 7
    else:
        id1 = 8
    return id1


def get_gt_elements(
    imname,
    eledir,
    simple=False
):
    elementLabel = pd.read_csv(os.path.join(eledir, imname), names=[
                               'bboxID', 'category', 'x', 'y'])
    elementCoords = []
    elementX = []
    elementY = []
    tmp = 0
    curName = ""

    boxes = []

    for row in elementLabel.iterrows():
        # row[1][0]: id
        # row[1][1]: category name
        # row[1][2]: x
        # row[1][3]: y

        # a new bbox
        if int(row[1][0]) > tmp:
            # Store the last bbox
            if tmp > 0:
                if (simple):
                    boxid = get_bboxid_by_name_9(curName)
                else:
                    boxid = get_bboxid_by_name(curName)
                box = BBox(curName, elementCoords[-1], boxid)
                boxes.append(box)
            tmp = int(row[1][0])
            curName = row[1][1].strip()

            elementCoords.append([])
            elementX.append([])
            elementY.append([])

        elementCoords[-1].append([int(row[1][2]), int(row[1][3])])
        if(curName != 'data'):
            elementX[-1].append(int(row[1][2]))
            elementY[-1].append(int(row[1][3]))

    boxid = get_bboxid_by_name(curName)
    box = BBox(curName, elementCoords[-1], boxid)
    boxes.append(box)
    return elementCoords, elementX, elementY, boxes


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

MARKERS = ['s', 'P', '*', 'X', 'p', 'v']
LINESTYLES = ['dotted', 'dashed', 'dashdot', '-', '--', '-.', ':', 'solid']

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_overall_image(
    imname, im, width, height, allfiles, excluded_subjects,
    outpath, elementX, elementY, boxes, length, out_postpend
) -> None:
    # Add the image to the path
    fig = plt.figure(figsize=(16, 16))

    plt.imshow(im)
    x_heat = []
    y_heat = []

    for subcsv in allfiles:
        if os.path.basename(subcsv)[:-4] in excluded_subjects:
            continue
        fixations = pd.read_csv(subcsv, header=None, names=[
                                'index', 'x', 'y', 'time'])

        x = fixations['x'].tolist()
        y = fixations['y'].tolist()
        duration = fixations['time'].tolist()
        length.append(len(x))

        x_heat.extend(x)
        y_heat.extend(y)

        plt.plot(x, y, '-ro', color='yellow', lw=1)
        # visualize the plot on image

        for i in range(len(x)):
            dur = duration[i]/10 if duration[i]/10 > 10 else 10
            dur = dur if dur < 20 else 20
            if(i == 0):
                plt.plot(x[i], y[i], '-o', ms=dur,
                         mfc=(0.1, 0.1, 1.0, 0.5))
            elif(i == len(x) - 1):
                plt.plot(x[i], y[i], '-o', ms=dur,
                         mfc=(0.5, 0.5, 0.5, 0.2), mec='#e9b96e')
            else:
                plt.plot(x[i], y[i], '-o', ms=dur,
                         mfc=(1.0, 0.0, 0.0, 0.5), mec='#e9b96e')
            plt.annotate(i, (x[i], y[i]), color=(
                1, 1, 0, 0.7), weight='bold', fontsize=5)

    for i in range(len(elementX)):
        plt.fill(elementX[i], elementY[i],
                 color=BUCKETCOLORS[boxes[i].id])

    plt.title(imname+'_total')
    fig.savefig('%s/%s_%s_sum.png' %
                (outpath, imname, out_postpend), dpi=fig.dpi)
    plt.cla()
    plt.clf()
    plt.close(fig)

    fm, hm = make_fixmap_and_heatmap(
        width, height, list(zip(x_heat, y_heat)))
    fm.close()
    hm.save('%s/%s_%s_sum_hm.png' % (outpath, imname, out_postpend))
    hm.close()

    plt.close('all')


def determine_correct_answers(
    df_questions, allfiles, imname
):
    imname = f"{imname}.png"
    correct, in_correct = [], []
    for file in allfiles:
        question_id = file.split(os.path.sep)[-2]
        subject = os.path.basename(file)[:-4]
        df = df_questions[(df_questions['subject'] == subject)
                          & (df_questions['image_name'] == imname)
                          & (df_questions['question_id'] == int(question_id))]
        try:
            assert(len(df) == 1)
            solution = df.iloc[0]['correct']
            answer = df.iloc[0]['subject_answer']
            if answer == solution:
                correct.append(file)
            else:
                in_correct.append(file)
        except AssertionError:
            print(question_id, subject, imname)
    return correct, in_correct


def compare_scanpaths_by_subject_answer(
    dataset_dir: str, baseline: bool
) -> None:
    excluded = pd.read_csv("dataset/excluded.csv")
    excluded_subjects = excluded['subject_id'].unique()
    if baseline:
        df_questions = pd.read_csv(f"{dataset_dir}/questions_baseline.csv")
        fixationsByVis = f"{dataset_dir}/BASELINES/csv_files/fixationsByVis"
    else:
        df_questions = pd.read_csv(f"{dataset_dir}/questions.csv")
        fixationsByVis = f"{dataset_dir}/eyetracking/csv_files/fixationsByVis"
    visqa_img = f"{dataset_dir}/merged/src"
    elem_dir = f"{dataset_dir}/element_labels"

    if baseline:
        outpath = os.path.join("../DATASET/VisQA/", 'plot_overall_baseline')
        os.makedirs(outpath, exist_ok=True)
    else:
        outpath = os.path.join("../DATASET/VisQA/", 'plot_overall')
        os.makedirs(outpath, exist_ok=True)

    print(outpath)
    length = []
    curfiles = natsorted(glob(fixationsByVis + '/*'))

    for id in trange(0, len(curfiles), 1):
        imname = os.path.basename(curfiles[id])
        # print(basepath)
        # imname, ext = os.path.splitext(basepath)
        elementCoords, elementX, elementY, boxes = get_gt_elements(
            imname, elem_dir)

        # load the raw image
        png = os.path.join(visqa_img, imname+'.png')
        jpg = os.path.join(visqa_img, imname+'.jpg')

        if os.path.isfile(png):
            path = png
        elif os.path.isfile(jpg):
            path = jpg

        with Image.open(path) as im:
            width, height = im.size  # original image size

            # Handle bbox Overlap
            id_map = np.full([width, height], 13)

            # Preserve bbox Overlap
            id_map = np.zeros((13, width, height))
            id_other = np.ones((width, height))

            for tt in range(12, -1, -1):
                for box in boxes:
                    if box.id == tt:
                        rr, cc = polygon(
                            np.array(box.coords)[:, 0],
                            np.array(box.coords)[:, 1],
                            (width, height))
                        id_map[tt, rr, cc] = 1
                        id_other[rr, cc] = 0

            allfiles = glob(os.path.join(
                curfiles[id], 'recall', '*', '*.csv'))

            correct, incorrect = determine_correct_answers(
                df_questions, allfiles, imname)

            plot_overall_image(
                imname, im, width, height, correct, excluded_subjects,
                outpath, elementX, elementY, boxes, length, "correct"
            )

            plot_overall_image(
                imname, im, width, height, incorrect, excluded_subjects,
                outpath, elementX, elementY, boxes, length, "incorrect"
            )


def calc_fixation_densities(dataset_dir: str, baseline: bool) -> None:
    """Calculates fixation densities and plots plot_overall

    :param dataset_dir: The dataset location.
    :type dataset_dir: str
    """
    excluded = pd.read_csv("dataset/excluded.csv")
    excluded_subjects = excluded['subject_id'].unique()
    if baseline:
        fixationsByVis = f"{dataset_dir}/BASELINES/csv_files/fixationsByVis"
    else:
        fixationsByVis = f"{dataset_dir}/eyetracking/csv_files/fixationsByVis"
    visqa_img = f"{dataset_dir}/merged/src"
    elem_dir = f"{dataset_dir}/element_labels"

    if baseline:
        outpath = os.path.join("../DATASET/VisQA/", 'plot_overall_baseline')
        os.makedirs(outpath, exist_ok=True)
    else:
        outpath = os.path.join("../DATASET/VisQA/", 'plot_overall')
        os.makedirs(outpath, exist_ok=True)

    length = []
    curfiles = natsorted(glob(fixationsByVis + '/*'))

    mds_bucket = np.zeros((14, 20))
    area = np.zeros((14))

    for id in trange(0, len(curfiles), 1):
        imname = os.path.basename(curfiles[id])
        # print(basepath)
        # imname, ext = os.path.splitext(basepath)
        elementCoords, elementX, elementY, boxes = get_gt_elements(
            imname, elem_dir)

        # load the raw image
        png = os.path.join(visqa_img, imname+'.png')
        jpg = os.path.join(visqa_img, imname+'.jpg')

        if os.path.isfile(png):
            path = png
        elif os.path.isfile(jpg):
            path = jpg

        with Image.open(path) as im:
            width, height = im.size  # original image size

            # Handle bbox Overlap
            id_map = np.full([width, height], 13)

            # Preserve bbox Overlap
            id_map = np.zeros((13, width, height))
            id_other = np.ones((width, height))

            for tt in range(12, -1, -1):
                for box in boxes:
                    if box.id == tt:
                        rr, cc = polygon(
                            np.array(box.coords)[:, 0],
                            np.array(box.coords)[:, 1],
                            (width, height))
                        # id_map[rr, cc] = tt
                        id_map[tt, rr, cc] = 1
                        id_other[rr, cc] = 0

            for tt in range(13):
                area[tt] += np.sum(id_map[tt] > 0)
            area[13] = np.sum(id_other > 0)

            allfiles = glob(os.path.join(curfiles[id], 'enc', '*.csv'))

            # Add the image to the path
            fig = plt.figure(figsize=(16, 16))

            plt.imshow(im)
            x_heat = []
            y_heat = []

            for subcsv in allfiles:
                if os.path.basename(subcsv)[:-4] in excluded_subjects:
                    continue
                fixations = pd.read_csv(subcsv, header=None, names=[
                                        'index', 'x', 'y', 'time'])

                x = fixations['x'].tolist()
                y = fixations['y'].tolist()
                duration = fixations['time'].tolist()
                length.append(len(x))

                x_heat.extend(x)
                y_heat.extend(y)

                accum_time = 0
                for row2 in fixations.iterrows():
                    accum_time += row2[1][3]
                    if (int(row2[0]) < 2):
                        continue
                    # If the fixation is inside the box,
                    # add one to the bucket counter
                    id2 = get_accum_bucket(accum_time)
                    for mm in range(13):
                        try:
                            mds_bucket[mm][id2] += id_map[
                                mm, int(row2[1][1] - 1), int(row2[1][2] - 1)]
                        except IndexError:
                            print(subcsv)
                            print(path)
                    # id1 = id_map[int(row2[1][1]),int(row2[1][2])]
                    # mds_bucket[id1][id2] += id
                    try:
                        mds_bucket[13][id2] += id_other[
                            int(row2[1][1] - 1), int(row2[1][2] - 1)]
                    except IndexError:
                        print(subcsv)
                        print(path)
                x.pop(0)
                y.pop(0)
                duration.pop(0)

                plt.plot(x, y, '-ro', color='yellow', lw=1)
                # visualize the plot on image

                for i in range(len(x)):
                    dur = duration[i]/10 if duration[i]/10 > 10 else 10
                    dur = dur if dur < 20 else 20
                    if(i == 0):
                        plt.plot(x[i], y[i], '-o', ms=dur,
                                 mfc=(0.1, 0.1, 1.0, 0.5))
                    elif(i == len(x) - 1):
                        plt.plot(x[i], y[i], '-o', ms=dur,
                                 mfc=(0.5, 0.5, 0.5, 0.2), mec='#e9b96e')
                    else:
                        plt.plot(x[i], y[i], '-o', ms=dur,
                                 mfc=(1.0, 0.0, 0.0, 0.5), mec='#e9b96e')
                    plt.annotate(i, (x[i], y[i]), color=(
                        1, 1, 0, 0.7), weight='bold', fontsize=5)

            for i in range(len(elementX)):
                plt.fill(elementX[i], elementY[i],
                         color=BUCKETCOLORS[boxes[i].id])

            plt.title(imname+'_total')
            fig.savefig('%s/%s_sum.png' % (outpath, imname), dpi=fig.dpi)
            plt.cla()
            plt.clf()
            plt.close(fig)

            fm, hm = make_fixmap_and_heatmap(
                width, height, list(zip(x_heat, y_heat)))
            fm.close()
            hm.save('%s/%s_sum_hm.png' % (outpath, imname))
            hm.close()

            plt.close('all')

    fp = open("mds_bucket.txt", "w")
    np.savetxt(fp, mds_bucket, fmt='%.2f')
    fp.close()

    fp = open("area.txt", "w")
    np.savetxt(fp, area, fmt='%.2f')
    fp.close()

    res = mds_bucket.copy()

    for i in range(np.shape(res)[1]):
        res[:, i] /= np.sum(res[:, i])
        res[:, i] = np.round(100*res[:, i], 2)

    fp = open("res.txt", "w")
    np.savetxt(fp, res, fmt='%.2f')
    fp.close()


def plot_overall(dataset_dir: str, baseline: bool) -> None:
    compare_scanpaths_by_subject_answer(dataset_dir, baseline)
    calc_fixation_densities(dataset_dir, baseline)


def vis_fixation_density_plot(path_to_csv: str) -> None:
    df_10k = pd.read_csv(path_to_csv)  # , index_col=0, header=None).T
    df = df_10k.T

    df = df.drop('Label/Time')
    df = df.dropna()

    # normalization
    column_maxes = df.max()
    df_max = column_maxes.max()
    df = df / df_max

    # Renaming and creating dataframe for Average patterns
    df = df.rename(columns={0: "Group 1", 4: "Group 2", 9: "Group 3"})
    df = df.transform(lambda x: x**0.5)
    df_line = df

    df_10k_orig = df_10k

    # Removing the average results from the main dataframe
    df_10k = df_10k.drop(0)
    df_10k = df_10k.drop(4)
    df_10k = df_10k.drop(9)
    df_10k = df_10k.sort_values(by=['Average'], ascending=False)
    x_list = df_10k['Label/Time'].to_list()
    df_10k.pop('Label/Time')

    list_of_cols = df_10k.columns
    df_10k = df_10k.T
    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(ncols=1, nrows=3)
    df_line = df_line.drop(['Average'], axis=0)
    df_line_g1 = df_line.iloc[:, 0:4]
    df_line_g2 = df_line.iloc[:, 4:8]
    df_line_g3 = df_line.iloc[:, 8:15]

    """
    Custom color map trials
    """
    # pal = sns.color_palette('rainbow',11)
    # pal = cmr.take_cmap_colors('rainbow', 12, return_fmt='hex')
    pal = ['#54478c', '#2c699a', '#048ba8', '#0db39e', '#16db93', '#83e377',
           '#b9e769', '#b9e769', '#efea5a', '#f1c453', '#f29e4c', '#f24c00']

    pal_group1 = [pal[8], pal[10], pal[2]]
    pal_group2 = [pal[0], pal[4], pal[6]]
    pal_group3 = [pal[1], pal[3], pal[5], pal[7], pal[9], pal[11]]

    linewidth_list_use = df_10k_orig['Fixation Density'].tolist()

    linewidth_list_use_g1 = linewidth_list_use[1:4]
    linewidth_list_use_g2 = linewidth_list_use[5:8]
    linewidth_list_use_g3 = linewidth_list_use[9:14]

    ax1 = pl.subplot(gs[0, 0])
    ax1 = df_line['Group 1'].plot(
        alpha=0.7, linewidth=6, color="grey", marker='.')
    df_line_g1 = df_line_g1.drop(['Group 1'], axis=1)
    df_line_g1_iter = df_line_g1.columns
    for i, j in enumerate(df_line_g1_iter):
        ax1 = df_line_g1[j].plot(alpha=0.9, linewidth=linewidth_list_use_g1[i]
                                 * 3.5, color=pal_group1[i], linestyle=LINESTYLES[i], marker=MARKERS[i])

    ax3 = pl.subplot(gs[2, 0], sharex=ax1)
    ax3 = df_line['Group 2'].plot(
        alpha=0.7, linewidth=6, color="grey", marker='.')
    df_line_g2 = df_line_g2.drop(['Group 2'], axis=1)
    df_line_g2_iter = df_line_g2.columns
    print(df_line_g2.columns)
    for i, j in enumerate(df_line_g2_iter):
        ax3 = df_line_g2[j].plot(alpha=0.9, linewidth=linewidth_list_use_g2[i]
                                 * 3.5, color=pal_group2[i], linestyle=LINESTYLES[i], marker=MARKERS[i])

    ax5 = pl.subplot(gs[1, 0], sharex=ax1)
    ax5 = df_line['Group 3'].plot(
        alpha=0.7, linewidth=6, color="grey", marker='.')
    df_line_g3 = df_line_g3.drop(['Group 3'], axis=1)
    df_line_g3_iter = df_line_g3.columns
    for i, j in enumerate(df_line_g3_iter):
        ax5 = df_line_g3[j].plot(alpha=0.9, linewidth=linewidth_list_use_g3[i]
                                 * 3.5, color=pal_group3[i], linestyle=LINESTYLES[i], marker=MARKERS[i])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax3.get_legend_handles_labels()
    handles3, labels3 = ax5.get_legend_handles_labels()

    labels1 = ['Average', '0', '1', '2']
    labels2 = ['Average', '4', '5', '6']
    labels3 = ['Average', '8', '9', '10', '11', '12']

    rules_g1 = {
        '0': 'Graphics',
        '1': 'Title',
        '2': 'Paragraph',
    }

    rules_g2 = {
        '4': 'Header Row',
        '5': 'Object',
        '6': 'Data',
    }

    rules_g3 = {
        '8': 'Annotation',
        '9': 'Axis',
        '10': 'Label',
        '11': 'Legend',
        '12': 'Source',
    }

    sample_list_1 = labels1
    labels_list_1 = [rules_g1.get(c, c) for c in sample_list_1]
    custom_order_1 = [0, 1, 2, 3]
    labels_list_custom_1 = [labels_list_1[i] for i in custom_order_1]
    handle_list_custom_1 = [handles1[i] for i in custom_order_1]

    sample_list_2 = labels2
    labels_list_2 = [rules_g2.get(c, c) for c in sample_list_2]
    custom_order_2 = [0, 1, 2, 3]
    labels_list_custom_2 = [labels_list_2[i] for i in custom_order_2]
    handle_list_custom_2 = [handles2[i] for i in custom_order_2]

    sample_list_3 = labels3
    labels_list_3 = [rules_g3.get(c, c) for c in sample_list_3]
    custom_order_3 = [0, 1, 2, 3, 4, 5]
    labels_list_custom_3 = [labels_list_3[i] for i in custom_order_3]
    handle_list_custom_3 = [handles3[i] for i in custom_order_3]

    legend = ax1.legend(frameon=False)
    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(1)
        ax1.legend(handles=handle_list_custom_1, loc='upper right', ncol=1, bbox_to_anchor=(
            1.0, 1.0), borderaxespad=0, labels=labels_list_custom_1)

    legend2 = ax3.legend(frameon=False)
    for legend_handle in legend2.legendHandles:
        legend_handle._legmarker.set_markersize(1)
        ax3.legend(handles=handle_list_custom_2, loc='upper right', ncol=1, bbox_to_anchor=(
            1.0, 1.0), borderaxespad=0, labels=labels_list_custom_2)

    legend3 = ax5.legend(frameon=False)
    for legend_handle in legend3.legendHandles:
        legend_handle._legmarker.set_markersize(1)
        ax5.legend(handles=handle_list_custom_3, loc='upper right', ncol=1, bbox_to_anchor=(
            1.0, 1.0), borderaxespad=0, labels=labels_list_custom_3)

    ax1.set_ylim([0, 1.05])
    ax3.set_ylim([0, 1.05])
    ax5.set_ylim([0, 1.05])

    list_xlabel_ticks = ['', '1', '', '2', '', '3', '', '4', '', '5', '', '6',
                         '', '7', '', '8', '', '9', '', '10']
    ax5.set_xlabel_ticks = list_xlabel_ticks
    ax5.set_xticks(range(0, len(list_xlabel_ticks)))
    ax5.set_xticklabels(list_xlabel_ticks)
    plt.setp(ax1.get_xticklabels(), rotation=0, va='top')

    ax1.tick_params(axis='y', which='major', pad=4)
    ax3.tick_params(axis='y', which='major', pad=2)
    ax5.tick_params(axis='y', which='major', pad=4)

    # ax1.get_xaxis().set_visible(False)
    # ax3.get_xaxis().set_visible(False)
    plt.minorticks_off()
    plt.setp(ax1.get_xticklabels())
    ax3.set_xlabel("Fixation timestamps (s)")
    ax3.set_ylabel("Element Fixation Density",
                   labelpad=10, position=(-0.5, 1.6))
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.subplots_adjust(hspace=.12)
    # plt.subplots_adjust(wspace=0.13)

    plt.savefig("Final_plot_figure.png", bbox_inches='tight', dpi=400)


def fixation_desnity_auc(path: str):
    df_10k = pd.read_csv(path)
    df = df_10k.T
    df = df.drop('Label/Time')
    df = df.dropna()
    df = df[[6, 10, 13]]
    df = df.rename(columns={6: "Inverse Boomerang Avg",
                   10: "Boomerang Avg", 13: "Increasing Avg"})

    df_10k = df_10k.drop(6)
    df_10k = df_10k.drop(10)
    df_10k = df_10k.drop(13)

    df_10k = df_10k.sort_values(by=['Average'], ascending=False)

    df_10k_orig = df_10k

    x_list = df_10k['Label/Time'].to_list()
    df_10k.pop('Label/Time')
    df_bar = pd.DataFrame()
    df_bar['average'] = df_10k_orig['Average'].tolist()
    df_bar['labels_name'] = x_list
    df_bar['labels'] = np.zeros(len(df_bar))
    df_bar['Fixation_density'] = df_10k_orig['Fixation Density'].tolist()
    df_10k.pop('Average')
    df_10k.pop('Fixation Density')

    # uncomment this line to remove the grids
    plt.style.use('seaborn')
    # use the below line to customize the grid
    sns.set_style("whitegrid", {'axes.grid': False})

    list_of_cols = df_10k.columns
    for i in range(len(list_of_cols)):
        time_s = list_of_cols[i]
        df_10k[time_s] = df_10k[time_s].sum() - df_10k[time_s]

    df_10k = df_10k.T
    fig = plt.figure(figsize=(10, 8.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1])
    ax1 = pl.subplot(gs[0, 0])
    ax = df_10k.plot.area(cmap='rocket', alpha=0.7, linewidth=1,
                          path_effects=[
                              pe.Stroke(linewidth=4, foreground='w'), pe.Normal()],
                          xticks=range(0, len(df_10k.index)), ax=ax1, stacked=True)
    handles, labels = ax.get_legend_handles_labels()
    # plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax1.set_xticklabels([])

    red_line = matplotlib.lines.Line2D([], [], color='red', markersize=500)
    blue_line = matplotlib.lines.Line2D([], [], color='blue', markersize=500)
    green_line = matplotlib.lines.Line2D([], [], color='green', markersize=500)
    inv_boomerang = [red_line]
    boomerang = [blue_line]
    increasing = [green_line]

    labels_list = ['Inverse Boomerang pattern',
                   'Boomerang pattern', 'Increasing pattern']

    handles_list = handles + inv_boomerang + boomerang + increasing
    labels_list = x_list + labels_list
    custom_order = [11, 0, 2, 3, 4, 6, 7, 12, 5, 9, 10, 13, 1, 8]
    labels_list_custom = [labels_list[i] for i in custom_order]
    handle_list_custom = [handles_list[i] for i in custom_order]

    ax.legend(handles=handle_list_custom, loc='upper right', ncol=1, bbox_to_anchor=(
        1.75, 1.25), borderaxespad=8, labels=labels_list_custom)
    ax2 = pl.subplot(gs[:, 1])
    our_custom_ticks = df_bar['Fixation_density'].tolist()
    sns.stripplot(data=df_bar, x="labels", y="Fixation_density",
                  hue="labels_name", ax=ax2, palette='rocket', s=10, jitter=False)
    ax2.get_legend().remove()
    ax2.set_ylabel('')
    ax2.set_xlabel('Labels')

    ax3 = pl.subplot(gs[1, 0])  # row 1, span all columns
    df_err = df-df.mean()
    df_err["Inverse Boomerang Avg"] = pd.to_numeric(
        df_err["Inverse Boomerang Avg"], downcast="float")
    df_err["Boomerang Avg"] = pd.to_numeric(
        df_err["Boomerang Avg"], downcast="float")
    df_err["Increasing Avg"] = pd.to_numeric(
        df_err["Increasing Avg"], downcast="float")

    df["Inverse Boomerang Avg"] = pd.to_numeric(
        df["Inverse Boomerang Avg"], downcast="float")
    df["Boomerang Avg"] = pd.to_numeric(df["Boomerang Avg"], downcast="float")
    df["Increasing Avg"] = pd.to_numeric(
        df["Increasing Avg"], downcast="float")

    ax3 = df['Inverse Boomerang Avg'].plot(alpha=8, linewidth=7, color="red", path_effects=[
                                           pe.Stroke(linewidth=8, foreground='white'), pe.Normal()])
    ax3 = df['Boomerang Avg'].plot(alpha=8, linewidth=5, color="blue", path_effects=[
                                   pe.Stroke(linewidth=6, foreground='white'), pe.Normal()])
    ax3 = df['Increasing Avg'].plot(alpha=8, linewidth=6, color="green", path_effects=[
                                    pe.Stroke(linewidth=7, foreground='white'), pe.Normal()])
    # plt.text(20, 9.20, 'Avg. boomerang pattern',color="blue",fontsize=10)
    # plt.text(20, 6.99, 'Avg. inverse boomerang pattern',color="red",fontsize=10)
    # plt.text(20, 12.45, 'Avg. increasing pattern',color= "green",fontsize=10)
    list_xlabel_ticks = df_10k_orig.columns

    df = df.drop('Average')
    df_err = df_err.drop('Average')

    plt.fill_between(list_xlabel_ticks, df['Inverse Boomerang Avg'] - df_err['Inverse Boomerang Avg'],
                     df['Inverse Boomerang Avg'] + df_err['Inverse Boomerang Avg'], alpha=0.5, color='red')
    plt.fill_between(list_xlabel_ticks, df['Boomerang Avg'] - df_err['Boomerang Avg'],
                     df['Boomerang Avg'] + df_err['Boomerang Avg'], alpha=0.5, color='blue')
    plt.fill_between(list_xlabel_ticks, df['Increasing Avg'] - df_err['Increasing Avg'],
                     df['Increasing Avg'] + df_err['Increasing Avg'], alpha=0.5, color='green')
    ax3.set_xticks(range(0, len(list_xlabel_ticks)))
    ax3.set_xlabel("Time in milli seconds")
    ax3.set_xticklabels(list_xlabel_ticks)
    plt.setp(ax3.get_xticklabels(), rotation=50, horizontalalignment='right')
    plt.ylabel("Relative accumulated fixation density over time",
               labelpad=10, fontsize=12, position=(5, 1.25))
    fig.suptitle("Fixation density - AUC plot", fontsize=18)
    plt.savefig("Desnity_AUC.png", dpi=400)


def auxiliary_statistics(df_10k_path, rate_list=[1.0, 0.231, 0.158, 0.445, 0.391, 0.557, 0.661, 0.189, 0.690, 0.246, 0.167]):

    sns.set_theme(style="whitegrid")

    # 11 Labels
    rate = np.array(rate_list)

    labels = np.array(["Annotation", "Axis", "Graphical Element", "Legend",
                       "Object", "Title", "Header Row", "Label", "Paragraph", "Source", "Data"])

    # 8 Labels

    # rate = np.array([0.579,0.510,0.564,1.000,0.364,0.967,0.830,0.329])
    # labels = np.array([ "Annotation", "Axis", "Graphical Element", "Legend", \
    # "Object", "Title & Header Row & Paragraph", "Label & Source", "Data"])

    df = pd.DataFrame({
        'Labels': labels,
        'Fixation Density': rate
    })

    plot_order = df.sort_values(
        by='Fixation Density', ascending=False).Labels.values

    fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=144)

    # Make the PairGrid
    sns.barplot(y="Labels", x="Fixation Density", data=df,
                order=plot_order, palette='rocket')
    ax.set_xlabel('')
    ax.set_ylabel('')

    sns.despine(top=True, right=True)

    plt.show()

    fig.savefig('FixationDensity.png')

    df_10k = pd.read_csv(df_10k_path)

    df = df_10k.T

    df = df.drop('Label/Time')
    df = df.dropna()

    # normalization
    column_maxes = df.max()
    df_max = column_maxes.max()
    df = df / df_max

    # Renaming and creating dataframe for Average patterns
    df = df.rename(columns={0: "Group 1", 3: "Group 2", 8: "Group 3"})
    df = df.transform(lambda x: x**0.5)
    df_line = df
    df = df[['Group 1', 'Group 2', 'Group 3']]
    df = df.drop(['Average'])

    print(df_10k)
    # Removing the average results from the main dataframe
    df_10k = df_10k.drop(0)
    df_10k = df_10k.drop(3)
    df_10k = df_10k.drop(8)
    df_10k_orig = df_10k
    x_list = df_10k['Label/Time'].to_list()
    df_10k.pop('Label/Time')

    print(x_list)
    from sklearn.cluster import KMeans
    df_10k = df_10k.drop(['Average'], axis=1)
    df_10k = df_10k.drop(['Fixation Density'], axis=1)
    data = df_10k.iloc[:20].to_numpy()
    for i in range(11):
        data[i, :] = data[i, :] / data[i, 1]

    estimator = KMeans(n_clusters=3)
    estimator.fit(data)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    print(label_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--vis_fixation_density_plot",
                        type=str, default=None)
    parser.add_argument("--fixation_desnity_auc",
                        type=bool, default=False)
    args = vars(parser.parse_args())
    if args['dataset_dir'] is not None:
        calc_fixation_densities(args['dataset_dir'])
    elif args['vis_fixation_density_plot']:
        vis_fixation_density_plot(args['vis_fixation_density_plot'])
    elif args['fixation_desnity_auc']:
        fixation_desnity_auc('Analysis/Densities/plot_fixation_densities.csv')
    else:
        auxiliary_statistics('Analysis/Densities/plot_fixation_densities.csv')
