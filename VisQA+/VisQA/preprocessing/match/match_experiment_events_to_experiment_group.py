from typing import Dict, Tuple

import json
import argparse
import pandas as pd
from collections import namedtuple


NUM_MSEC_CROSS = 750
NUM_MSEC_IMAGE = 10000
NUM_MSCEC_RECOGNITION_IMAGE = 2000


def get_image_file_name_from_image_path(image_path: str) -> str:
    """
    Given a path like ../bla/fo/boo/image.png returns image.png.

    :param image_path: Path to the image.
    :returns: The image file name.
    """
    return image_path.split("\\")[-1]


def match_experiment_events_to_experiment_group(
    df_events: pd.DataFrame,
    experiment_group: Dict,
    baseline: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Takes a list of events from a complete experiment run and matches it to
    the experiment description dict coming from the appropiate description JSON
    file. Returns the matches as a dataframe.

    :param df_events: Dataframe containing the events of a complete experiment_group
        recording. Columns: event_name, target, target_id, target_name,
            event_stage, timestamp
    :type df_events: pd.DataFrame
    :param experiment_group: Experiment group dictionary.
    :type experiment_group: Dict
    :return: Returns two dataframes one containing the viewed images,
        their length etc and one similiar but with questions related
        to images instead given a experiment events dataframe and the
        experiment description. Also returns a DF containing fixations
        on the verification crosses. The exact return description is:
        (
            DF('image_name', 'kind','image_flag', 'stage','start_time',
                'end_time'),
            DF('image_name', 'kind', 'image_flag', 'stage', 'start_time',
                'end_time', 'question_id', 'question', 'A', 'B', 'C',
                'D', 'correct')
            DF(see first)
        )
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    df_question_views = []
    df_image_views = []
    df_cross_views = []

    tmp_question_views = []
    question_start = None
    last_row = None

    for row in df_events.itertuples(name='event'):
        if last_row is None:
            last_row = row
            continue

        if (row.event_name != "QA button clicked"
                and row.event_name != "next button clicked"
                and row.event_name != "Entering recognition stage"):
            last_row = row
            continue

        if row.target_name == "task" and last_row.target_name != "question":
            id = int(row.target_id)
            group = experiment_group[int(id)]
            image_path = group['image']
            image_flag = group['flag']
            image = get_image_file_name_from_image_path(image_path)
            start_time = int(last_row.timestamp) + NUM_MSEC_CROSS
            end_time = int(last_row.timestamp) + NUM_MSEC_CROSS + \
                (NUM_MSEC_IMAGE if row.event_stage ==
                 "recall" and not baseline else NUM_MSCEC_RECOGNITION_IMAGE)
            df_cross_views.append(("cross_before_" + image, "view",
                                   image_flag, row.event_stage,
                                   int(last_row.timestamp),
                                   int(last_row.timestamp) + NUM_MSEC_CROSS))
            df_image_views.append((image, 'view', image_flag,
                                   'enc' if row.event_stage == 'recall' else row.event_stage,
                                   start_time, end_time))
            last_row = row
        elif (row.target_name == "question"
              and last_row.target_name != "question"):
            question_start = last_row
            tmp_question_views.append(row)
            last_row = row
        elif (row.target_name == "question"
              and last_row.target_name == "question"):
            tmp_question_views.append(row)
            last_row = row
        elif (row.target_name == "task"
              and last_row.target_name == "question"):
            event = namedtuple(
                'Event',
                'Index event_name target target_id target_name event_stage timestamp'
            )
            last_question = event(row.Index, last_row.event_name,
                                  'question No5', 5, last_row.target_name,
                                  last_row.event_stage, last_row.timestamp)
            tmp_question_views.append(last_question)
            img_id = int(row.target_id)
            group = experiment_group[int(img_id)]
            image_path = group['image']
            image_flag = group['flag']
            image = get_image_file_name_from_image_path(image_path)

            for i in range(len(tmp_question_views)):
                question_view = tmp_question_views[i]
                question_id = int(question_view.target_id)
                question = group['QA'][f'Q{question_id}']
                question_str = question['question']
                answer_a = question['A']
                answer_b = question['B']
                answer_c = question['C']
                answer_d = question['D']
                correct_answer = question['answer']
                if i == 0:
                    start_time = int(question_start.timestamp)
                else:
                    start_time = int(tmp_question_views[i-1].timestamp)
                end_time = int(question_view.timestamp)
                df_question_views.append(
                    (image, 'question', image_flag, question_view.event_stage,
                     start_time, end_time, question_id, question_str, answer_a,
                     answer_b, answer_c, answer_d, correct_answer))
            last_row = row
            tmp_question_views = []
            question_start = None

    df_question_views = pd.DataFrame.from_records(
        df_question_views,
        columns=['image_name', 'kind', 'image_flag', 'stage', 'start_time',
                 'end_time', 'question_id', 'question', 'A', 'B', 'C', 'D',
                 'correct']
    )
    df_cross_views = pd.DataFrame.from_records(
        df_cross_views,
        columns=['image_name', 'kind', 'image_flag',
                 'stage', 'start_time', 'end_time']
    )
    df_image_views = pd.DataFrame.from_records(
        df_image_views,
        columns=['image_name', 'kind', 'image_flag',
                 'stage', 'start_time', 'end_time']
    )
    df_image_views['question_id'] = -1
    df_question_image_views = df_question_views[
        ['image_name', 'kind', 'image_flag', 'stage',
            'start_time', 'end_time', 'question_id']
    ]
    return pd.concat([df_image_views, df_question_image_views]), df_question_views, df_cross_views


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_csv", type=str, default=None)
    parser.add_argument("--experiment_group_json", type=str, default=None)
    args = vars(parser.parse_args())

    if (args['events_csv'] is not None
            and args['experiment_group_json'] is not None):
        events = pd.read_csv(args['events_csv'])
        experiment_group = json.load(args['experiment_group_json'])
        match_experiment_events_to_experiment_group(events, experiment_group)
