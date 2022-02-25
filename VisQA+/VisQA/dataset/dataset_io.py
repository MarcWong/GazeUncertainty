from typing import Dict

import os
import json
import pandas as pd
import numpy as np
import sys


from pathlib import Path

REL_FIX_BY_VIS_PATH = "csv_files/fixationsByVis"
REL_STRINGS_PATH = "csv_files/strings"


def write_csv_fixations_by_vis_for_subject(
    subject_id: str,
    dataset_dir: str,
    df_user_fixations: pd.DataFrame,
    baseline: bool = False
) -> None:
    """Writes the fixations by vis to the filesystem just as in the
    Massvis dataset.

    :param subject_id: The subject id to name the files with.
    :type subject_id: str
    :param dataset_dir: The root to the dataset.
    :type dataset_dir: str
    :param df_user_fixations: The subjects fixations.
    :type df_user_fixations: pd.DataFrame
    :param baseline: Wether the baseline dataset should be processed.
    :type baseline: bool
    """
    if baseline:
        fix_by_vis_path = f"{dataset_dir}/BASELINES/{REL_FIX_BY_VIS_PATH}"
    else:
        fix_by_vis_path = f"{dataset_dir}/eyetracking/{REL_FIX_BY_VIS_PATH}"

    df = df_user_fixations[['axp', 'ayp', 'gxp', 'gyp',
                            'dur', 'file', 'stage', 'question_id']]
    df.axp = df.axp.round(2)
    df.ayp = df.ayp.round(2)

    # https://stackoverflow.com/questions/53316471/pandas-dataframes-to-csv-truncates-long-values
    df['gxp'] = df['gxp'].apply(lambda x: np.array2string(x, precision=2, separator=';', threshold=sys.maxsize, max_line_width=10000000, floatmode='fixed'))
    df['gyp'] = df['gyp'].apply(lambda y: np.array2string(y, precision=2, separator=';', threshold=sys.maxsize, max_line_width=10000000, floatmode='fixed'))

    for _, group in df.groupby(['file', 'stage', 'question_id']):
        stage = group.iloc[0]['stage']
        file = group.iloc[0]['file']
        question_id = group.iloc[0]['question_id']
        group.drop('file', axis=1, inplace=True)
        group.drop('stage', axis=1, inplace=True)
        group.drop('question_id', axis=1, inplace=True)
        group = group[['axp', 'ayp', 'dur', 'gxp', 'gyp']]
        group.reset_index(drop=True, inplace=True)

        if question_id == -1:
            Path(f"{fix_by_vis_path}/{file}/{stage}")\
                .mkdir(parents=True, exist_ok=True)
            group.to_csv(
                f"{fix_by_vis_path}/{file}/{stage}/{subject_id}.csv",
                header=False)
        else:
            Path(f"{fix_by_vis_path}/{file}/{stage}/{question_id}")\
                .mkdir(parents=True, exist_ok=True)
            group.to_csv(
                f"{fix_by_vis_path}/{file}/{stage}/{question_id}/{subject_id}.csv",
                header=False)


def convert_desc(desc: pd.Series):
    desc = desc.lower()
    if "annotation" in desc:
        return "D"
    elif "axis" in desc:
        return "X"
    elif "graphic" in desc:
        return "G"
    elif "legend" in desc:
        return "L"
    elif "object" in desc:
        return "O"
    elif "title" in desc:
        return "T"
    elif "text" in desc:  # S for Source text
        return "S"
    elif "data" in desc:
        return "D"
    else:
        print("Error!")
        print(desc)
        return "!"


def write_csv_fixations_by_strings(
    subject_id: str,
    dataset_dir: str,
    df_user_fixations: pd.DataFrame,
    baseline: bool = False
) -> None:
    """Writes a fixations element string by converting the
    looked at elment labels to single letters.

    :param subject_id: The subject id to write the file with.
    :type subject_id: str
    :param dataset_dir: The dataset root.
    :type dataset_dir: str
    :param df_user_fixations: The users fixations on elements.
    :type df_user_fixations: pd.DataFrame
    :param baseline: Wether the baseline dataset should be processed.
    :type baseline: bool
    """
    if baseline:
        strings_path = f"{dataset_dir}/BASELINES/{REL_STRINGS_PATH}"
    else:
        strings_path = f"{dataset_dir}/eyetracking/{REL_STRINGS_PATH}"

    df = df_user_fixations.dropna()
    df['desc'] = df['desc'].apply(lambda x: 'data' if 'data' in x else x)
    df['desc'] = df['desc'].apply(lambda x: 'object' if 'object' in x else x)
    df['desc'] = df['desc'].apply(convert_desc)

    for _, group in df.groupby(['file', 'stage', 'question_id']):
        stage = group.iloc[0]['stage']
        file = group.iloc[0]['file']
        question_id = group.iloc[0]['question_id']
        group.reset_index(drop=True, inplace=True)
        sequence = "".join(group['desc'].tolist())

        if question_id == -1:
            Path(f"{strings_path}/{subject_id}/{stage}")\
                .mkdir(parents=True, exist_ok=True)
            strings_txt = open(
                f"{strings_path}/{subject_id}/{stage}/{file}.txt", "w")
            strings_txt.write(sequence)
            strings_txt.close()
        else:
            Path(f"{strings_path}/{subject_id}/{stage}/{question_id}")\
                .mkdir(parents=True, exist_ok=True)
            strings_txt = open(
                f"{strings_path}/{subject_id}/{stage}/{question_id}/{file}.txt", "w")
            strings_txt.write(sequence)
            strings_txt.close()


def save_submit(submit: Dict, data_dir: str = os.getcwd() + "/dataset") -> str:
    """
    Saves submit data as json to our dataset structure.

    :param submit: Submit object to save.
    :returns: Path to the saved object.
    """
    assignement_id = submit["assignmentId"]
    worker_id = submit["workerId"]

    dir = Path(f'{data_dir}/experiments')
    dir.mkdir(parents=True, exist_ok=True)
    out = f'{str(dir)}/submit_ass{assignement_id}_w{worker_id}.json'
    with open(out, 'w') as fp:
        json.dump(submit, fp)
    return out
