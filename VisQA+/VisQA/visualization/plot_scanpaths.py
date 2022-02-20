import gc
import os
import re
import argparse
from typing import List
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from natsort import natsorted
from glob import glob
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from visualization.utils.visualization_utils import plot_visualization,\
    plot_img

matplotlib.use('agg')


def plot_scanpath_on_image(
    image_path: str,
    df_fixations_on_image:
    pd.DataFrame, out_path: str,
    threshold: int = None,
    DPI: int = 100
) -> None:
    """Plots the scanpath of a single subject on a single given
    image.

    :param image_path: The path to the image.
    :type image_path: str
    :param df_fixations_on_image: Dataframe containing the fixations
        on the image.
    :type df_fixations_on_image: pd.DataFrame
    :param out_path: Where to write the scanpath.
    :type out_path: str
    :param threshold: Minimal length of fixation, defaults to None
    :type threshold: int, optional
    :param DPI: DPI to use, defaults to 100
    :type DPI: int, optional
    """
    with Image.open(image_path) as img:
        if threshold is not None:
            df_fixations_on_image = \
                df_fixations_on_image[df_fixations_on_image['dur']
                                      >= threshold]

        x = df_fixations_on_image['axp'].tolist()
        y = df_fixations_on_image['ayp'].tolist()
        duration = df_fixations_on_image['dur'].tolist()
        fig = plot_img(img, DPI)

        plot_visualization(x, y, duration)
        fig.savefig(out_path, dpi=DPI)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        img.close()
        del fig


def plot_scanpaths_for_subject_verification(
    subject_dir: str,
    df_cross_fixations: pd.DataFrame,
    cross_img_path: str
) -> None:
    """Plot scanpaths on verification crosses.

    :param subject_dir: Dir of the subject.
    :type subject_dir: str
    :param df_cross_fixations: Dataframe containing fixation on verification
        crosses.
    :type df_cross_fixations: pd.DataFrame
    :param cross_img_path: Path to the image containing the verification cross.
    :type cross_img_path: str
    """
    image_names = list(set(df_cross_fixations['image_name'].tolist()))
    for image_name in image_names:
        plot_scanpath_on_image(cross_img_path,
                               df_cross_fixations[df_cross_fixations['image_name']
                                                  == image_name],
                               f"{subject_dir}/{image_name}_cross.png")


# @profile
def plot_scanpaths_for_subject(
    subject_path: str,
    image_dir: str,
    df_fixations: pd.DataFrame,
    threshold: int = None
) -> None:
    """Plot scan paths for subjects only for recall images.

    :param subject_path: Path to the subject for output.
    :type subject_path: str
    :param image_dir: Path to all images in the dataset.csv
    :type image_dir: str
    :param df_fixations: Dataframe containing all fixations of the user.
    :type df_fixations: pd.DataFrame
    :param threshold: Minimum fixation length, defaults to None
    :type threshold: int, optional
    """
    os.makedirs(f"{subject_path}/scanpaths", exist_ok=True)
    df_fixations_on_stage = df_fixations[df_fixations['stage'] == 'recall']
    image_names = list(set(df_fixations['image_name'].tolist()))

    for image_name in image_names:
        plot_scanpath_on_image(
            f"{image_dir}/{image_name}",
            df_fixations_on_stage[(df_fixations_on_stage['image_name'] == image_name)
                                  & (df_fixations_on_stage['stage'] == 'recall')],
            f"{subject_path}/scanpaths/scanpath_{image_name}",
            threshold)
    del df_fixations_on_stage


def make_fixmap_and_heatmap(w, h, coords, sigma=19):
    xs = tuple([elt[0] for elt in coords])
    ys = tuple([elt[1] for elt in coords])
    bitmap = np.zeros((w, h))
    fixations = np.zeros((w, h))
    for c in coords:
        x, y = int(c[0]), int(c[1])
        if x < w and y < h:
            fixations[x, y] += 1
            bitmap[x, y] = 1
    heatmap = ndimage.filters.gaussian_filter(fixations, [sigma, sigma])
    heatmap = 255*heatmap/float(np.max(heatmap))

    fix_img = Image.fromarray(np.uint8(np.transpose(bitmap)), "L")
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap)), "L")

    return fix_img, heatmap_img


def plot_scanpaths_for_subjects(
    fixationsByVis: str,
    images_dir: str,
    subjects_path: List[str],
    eyetracking_results_dir: str,
    baseline: bool,
    DPI: int = 144,
) -> None:
    """Plots all scanpaths by reading fixationsByVis so its
    not subject first but fixation first (in contrast to the
    function above).

    :param fixationsByVis: Directory containing fixations.
    :type fixationsByVis: str
    :param images_dir: Directory containing all images.
    :type images_dir: str
    :param subjects_path: The path to the subjects.
    :type subjects_path: List[str]
    :param eyetracking_results_dir: Where eyetreacking results are stored.
    :type subjects_path: str
    :param DPI: DPI to use, defaults to 144
    :type DPI: int, optional
    """
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
            w, h = im.size

            allfiles = natsorted(
                glob(os.path.join(fixationsByVis, fixation, 'enc', '*.csv')))

            for subcsv in allfiles:
                subject_id = os.path.basename(subcsv)[:-4]
                try:
                    subject_path = next(
                        path for path in subjects_path if subject_id in os.path.basename(path))
                except StopIteration:
                    group = [int(s)
                             for s in re.findall(r'\d+', subject_id)][0]
                    subject_path = os.path.abspath(
                        f"{eyetracking_results_dir}/G{group}/{subject_id}")
                    os.makedirs(subject_path, exist_ok=True)
                    subjects_path.append(subject_path)
                    print(
                        f"WARN: Subject {subject_id} not part of subjects_path.\
                        A path has been created.")
                outpath = f"{subject_path}/scanpaths"
                os.makedirs(outpath, exist_ok=True)

                fixations = pd.read_csv(subcsv, header=None)
                x = []
                y = []
                duration = []
                for row in fixations.iterrows():
                    x.append(row[1][1])
                    y.append(row[1][2])
                    duration.append(row[1][3])

                fm, hm = make_fixmap_and_heatmap(w, h, list(zip(x, y)))
                outpath_fm = f"{subject_path}/fixation_maps"
                os.makedirs(outpath_fm, exist_ok=True)
                fm.save(os.path.join(outpath_fm, imname + '.png'))
                fm.close()
                outpath_hm = f"{subject_path}/heat_maps"
                os.makedirs(outpath_hm, exist_ok=True)
                hm.save(os.path.join(outpath_hm, imname + '.png'))
                hm.close()

                fig = plot_img(im, DPI)

                plot_visualization(x, y, duration)

                fig.savefig('%s/%s%s.png' %
                            (outpath, imname, subject_id), dpi=DPI)
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
            im.close()
        gc.collect()

        with Image.open(path) as im:
            w, h = im.size

            allfiles = natsorted(
                glob(os.path.join(fixationsByVis, fixation, 'recognition', '*.csv')))

            for subcsv in allfiles:
                subject_id = os.path.basename(subcsv)[:-4]
                try:
                    subject_path = next(
                        path for path in subjects_path if subject_id in os.path.basename(path))
                except StopIteration:
                    group = [int(s)
                             for s in re.findall(r'\d+', subject_id)][0]
                    subject_path = os.path.abspath(
                        f"{eyetracking_results_dir}/G{group}/{subject_id}")
                    os.makedirs(subject_path, exist_ok=True)
                    subjects_path.append(subject_path)
                    print(
                        f"WARN: Subject {subject_id} not part of subjects_path.\
                        A path has been created.")
                outpath = f"{subject_path}/scanpaths_recognition"
                os.makedirs(outpath, exist_ok=True)

                fixations = pd.read_csv(subcsv, header=None)
                x = []
                y = []
                duration = []
                for row in fixations.iterrows():
                    x.append(row[1][1])
                    y.append(row[1][2])
                    duration.append(row[1][3])

                fig = plot_img(im, DPI)

                plot_visualization(x, y, duration)

                fig.savefig('%s/%s%s.png' %
                            (outpath, imname, subject_id), dpi=DPI)
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
            im.close()
        gc.collect()

        with Image.open(path) as im:

            allfiles = natsorted(
                glob(os.path.join(
                    fixationsByVis, fixation, 'recall', '*', '*.csv')))

            for subcsv in allfiles:
                subject_id = os.path.basename(subcsv)[:-4]
                question_id = os.path.split(os.path.split(subcsv)[0])[1]
                try:
                    subject_path = next(
                        path for path in subjects_path if subject_id in os.path.basename(path))
                except StopIteration:
                    group = [int(s)
                             for s in re.findall(r'\d+', subject_id)][0]
                    subject_path = os.path.abspath(
                        f"{eyetracking_results_dir}/G{group}/{subject_id}")
                    os.makedirs(subject_path, exist_ok=True)
                    subjects_path.append(subject_path)
                    print(
                        f"WARN: Subject {subject_id} not part of subjects_path.\
                        A path has been created.")
                outpath = f"{subject_path}/scanpaths_blurred/{question_id}"
                os.makedirs(outpath, exist_ok=True)

                fixations = pd.read_csv(subcsv, header=None)
                x = []
                y = []
                duration = []
                for row in fixations.iterrows():
                    x.append(row[1][1])
                    y.append(row[1][2])
                    duration.append(row[1][3])

                fm, hm = make_fixmap_and_heatmap(w, h, list(zip(x, y)))
                outpath_fm = f"{subject_path}/scanpaths_blurred/{question_id}/fixation_maps"
                os.makedirs(outpath_fm, exist_ok=True)
                fm.save(os.path.join(outpath_fm, imname + '.png'))
                fm.close()
                outpath_hm = f"{subject_path}/scanpaths_blurred/{question_id}/heat_maps"
                os.makedirs(outpath_hm, exist_ok=True)
                hm.save(os.path.join(outpath_hm, imname + '.png'))
                hm.close()

                fig = plot_img(im, DPI)
                plot_visualization(x, y, duration)

                fig.savefig('%s/%s%s.png' %
                            (outpath, imname, subject_id), dpi=DPI)
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
            im.close()
        gc.collect()
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--fixations_csv", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--cross_img_path", type=str, default=None)
    args = vars(parser.parse_args())

    if args["cross_img_path"] is not None:
        df_fixations = pd.read_csv(args['fixations_csv'])
        plot_scanpaths_for_subject_verification(
            args['subject_path'], df_fixations, args['cross_img_path'])
    else:
        df_fixations = pd.read_csv(args['fixations_csv'])
        plot_scanpaths_for_subject(
            args['subject_path'],
            args['image_dir'],
            df_fixations,
            args['threshold'])
