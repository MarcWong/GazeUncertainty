from typing import List

import os
import gc
import argparse
import pandas as pd

from tqdm import tqdm
from glob import glob

from preprocessing.parser.parse_element_labels import parse_element_labels
from preprocessing.preprocess import preprocess_subject
from dataset.dataset_io import write_csv_fixations_by_vis_for_subject,\
    write_csv_fixations_by_strings

from visualization.fixation_densities import plot_overall
from visualization.plot_scanpaths import plot_scanpaths_for_subjects
from visualization.text_saliency import visualize_text_saliency
from analysis.analysis import analyse_subject


def preprocess_single_subject_out(
    df_eye_fixations_to_elements,
    subject_id,
    dataset_dir,
    baseline
) -> None:
    """Writes all CSVs for a single subjet.

    :param df_eye_fixations_to_elements: Eye fixations for subject.
    :type df_eye_fixations_to_elements: pd.DataFrame
    :param df_question_views: Questions views for subject.
    :type df_question_views: pd.DataFrame
    :param path: Output path
    :type path: str
    :param subject_id: The subject id.
    :type subject_id: str
    :param dataset_dir: The path to the dataset.
    :type dataset_dir: str
    :param baseline: Wether baseline dataset to be processed
    :type baseline: bool
    """
    write_csv_fixations_by_vis_for_subject(
        subject_id, dataset_dir,  df_eye_fixations_to_elements, baseline)
    write_csv_fixations_by_strings(
        subject_id, dataset_dir, df_eye_fixations_to_elements, baseline)


def preprocess_all(
    dataset_dir: str,
    subjects_path: List[str],
    elem_labels_dir: str,
    subject_files_dir: str,
    html_subject_files_dir: str,
    baseline: bool,
    internal_time_to_unix_csv: str = "dataset/internal_to_unix_mapping.csv"
) -> pd.DataFrame:
    """Preprocesses all subjects.

    :param dataset_dir: Directory of the dataset.
    :type dataset_dir: str
    :param subjects_path: Path to subjects eyetracking results.
    :type subjects_path: List[str]
    :param elem_labels_dir: Directory where labels are stored.
    :type elem_labels_dir: str
    :param subject_files_dir: Subject files dir in VisQA.
    :type subject_files_dir: str
    :param baseline: Wether to use baseline dataset.
    :type baseline: bool
    :param internal_time_to_unix_csv: CSV for internal eye tracking timestamp.
    :type internal_time_to_unix_csv: str
    :rtype: pd.DataFrame
    """
    dataset = []
    df_internal_time_to_unix = pd.read_csv(
        internal_time_to_unix_csv, dtype={'verified': 'str'})
    df_internal_time_to_unix = df_internal_time_to_unix\
        .set_index(['subject'])

    df_element_labels = parse_element_labels(elem_labels_dir)

    print("Preprocessing:")
    for path in tqdm(subjects_path):
        subject_id = os.path.basename(path)

        try:
            mapping = df_internal_time_to_unix.loc[subject_id]
        except KeyError:
            print(
                f"Warn subject {subject_id} was skipped as no mapping exists.")
            continue

        if mapping.verified != "True":
            continue

        time_unix = int(mapping.timestamp_unix)
        time_main_content = int(mapping.timestampMainMessage_content)
        time_internal_unix = int(mapping.timestamp_internal)
        time_internal_main = int(mapping.timestamp_MainMessage)
        group = int(mapping.group)
        good_eye = str(mapping.good_eye)

        asc_file = f"{path}/{subject_id}.asc"
        experiment_desc = f"{subject_files_dir}/subject_file_0_{group}.json"
        experiment_desc_html = f"{html_subject_files_dir}/subject_file_0_{group}.json"

        debugout = glob(f"{path}/Images/debugout*.txt")
        assert(len(debugout) == 1)
        debugout_file = debugout[0]

        df_eye_fixations_to_elements, df_question_views, user_data\
            = preprocess_subject(
                group, df_element_labels, asc_file, debugout_file,
                experiment_desc, experiment_desc_html, dataset_dir, time_unix,
                time_internal_unix, time_internal_main, time_main_content,
                good_eye, baseline)
        df_eye_fixations_to_elements['subject'] = subject_id
        df_question_views['subject'] = subject_id

        preprocess_single_subject_out(
            df_eye_fixations_to_elements,
            subject_id,
            dataset_dir,
            baseline
        )

        dataset.append(df_eye_fixations_to_elements)

    df_dataset = pd.concat(dataset, axis=0)

    if baseline:
        df_dataset_path = f"{dataset_dir}/dataset_baseline.csv"
    else:
        df_dataset_path = f"{dataset_dir}/dataset.csv"

    df_dataset.to_csv(df_dataset_path, index=False)

    return df_dataset


def main(
    dataset_dir: str = None,
    eyetracking_results_dir: str = None,
    elem_labels_dir: str = None,
    images_dir: str = None,
    subject_files_dir: str = None,
    html_subject_files_dir: str = None,
    dataset_csv_path: str = None,
    analyze: bool = False,
    baseline: bool = False,
    internal_time_to_unix_csv: str = "dataset/internal_to_unix_mapping.csv"
) -> int:
    """
    Start the processing for every participant and write out the respective
    results.

    :param dataset_dir: Directory of the dataset.
    :type dataset_dir: str
    :param eyetracking_results_dir: Dir containing eye tracking results.
    :type eyetracking_results_dir: str
    :param elem_labels_dir: Directory where labels are stored.
    :type elem_labels_dir: str
    :param images_dir: Dir containing all relevant graphs.
    :type images_dir: str
    :param subject_files_dir: Subject files dir in VisQA.
    :type subject_files_dir: str
    :param internal_time_to_unix_csv: CSV for internal eye tracking timestamp.
    :type internal_time_to_unix_csv: str
    :param dataset_csv_path: The result of a previous run, is used as an
        indicator for skipping preprocessing. USE WITH CAUTION: THIS ASSUMES A
        CORRECT DATASET DIR FROM A PREVIOUS RUN (INCLUDING OTHER OUTPUTS)!
    :type dataset_csv_path: str
    :return: Exit code.
    :rtype: int
    """
    subjects_path = glob(os.path.join(
        f"{eyetracking_results_dir}/*/", "*", ""))
    subjects_path = [os.path.abspath(path) for path in subjects_path]

    if dataset_csv_path is not None:
        df_dataset = pd.read_csv(dataset_csv_path)
    else:
        df_dataset = preprocess_all(
            dataset_dir, subjects_path, elem_labels_dir,
            subject_files_dir, html_subject_files_dir, baseline,
            internal_time_to_unix_csv)

    if not analyze:
        return 0

    ##############################################################
    # Automatic plotting and basic calculations, contains:
    # 1) Scan path plotting
    # 2) Basic refixations calculations.
    # 3) Fixation density calculation and plot_overal.
    ##############################################################
    print("Visualizing:")
    if baseline:
        fixationsByVis = f"{dataset_dir}/BASELINES/csv_files/fixationsByVis"
        visualize_text_saliency(subjects_path)
    else:
        fixationsByVis = f"{dataset_dir}/eyetracking/csv_files/fixationsByVis"
        visualize_text_saliency(subjects_path)
    plot_overall(dataset_dir, baseline)
    plot_scanpaths_for_subjects(
        fixationsByVis, images_dir, subjects_path, eyetracking_results_dir,
        baseline)

    print("Analyzing:")
    subjects = df_dataset['subject'].unique()
    for subject_id in tqdm(subjects):
        df_user = df_dataset[df_dataset['subject'] == subject_id]
        df_refixations_user = analyse_subject(df_user)
        candidates = list(
            filter(
                lambda x: subject_id == os.path.basename(os.path.normpath(x)),
                subjects_path)
        )
        assert(len(candidates) == 1)
        path = candidates[0]
        df_refixations_user.to_csv(
            f"{path}/{subject_id}_refixations.csv", index=False)

        del df_user
        del df_refixations_user
        gc.collect()

    return 0


"""
Call looks like this (adjust for your VisQA download location):

python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/

or if you already have the dataset.csv provided you can skip preproccesing:

python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/
    --dataset_csv SOME/PATH/DATASET/VisQA/dataset.csv

Dataset_dir is still necessary for later output.

Additionally, --analyze True will run some automatically runable analysis,
i.e. a call this if you want every output we can provide:

python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/
    --analyze True

You can also specify all paths yourself: See below for options.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument("--eye_tracking_results", type=str, default=None)
    parser.add_argument("--elem_labels_dir", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--subject_files_dir", type=str, default=None)
    parser.add_argument("--subject_files_html_dir", type=str, default=None)
    parser.add_argument("--dataset_csv", type=str, default=None)
    parser.add_argument("--analyze", type=bool, default=False)
    args = vars(parser.parse_args())

    if (args['dataset_dir'] is not None
        and (args['eye_tracking_results'] is None
        or args['elem_labels_dir'] is None
        or args['images_dir'] is None
        or args['subject_files_dir'] is None
             or args['internal_time_to_unix_csv'] is None)):
        d = args['dataset_dir']
        if args['baseline']:
            main(d,
                 f"{d}/BASELINES/Results",
                 f"{d}/element_labels",
                 f"{d}/merged/src",
                 f"{d}/subject_files",
                 f"{d}/subject_files_html_baseline",
                 args['dataset_csv'],
                 args['analyze'],
                 args['baseline'],
                 "dataset/internal_to_unix_mapping_baseline.csv")
        else:
            main(d,
                 f"{d}/eyetracking/Results",
                 f"{d}/element_labels",
                 f"{d}/merged/src",
                 f"{d}/subject_files",
                 f"{d}/subject_files_html",
                 args['dataset_csv'],
                 args['analyze'],
                 args['baseline'])
    elif args['dataset_dir'] is not None:
        main(args['dataset_dir'],
             args['eye_tracking_results'],
             args['elem_labels_dir'],
             args['images_dir'],
             args['subject_files_dir'],
             args['subject_files_html_dir'],
             args['dataset_csv'],
             args['analyze'],
             args['baseline'])
    else:
        print("WARN: You need to atleast specify --dataset_dir.")
