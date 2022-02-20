import json
import argparse
import pandas as pd

from typing import Dict, List, Any, Tuple

from preprocessing.filter.filter_invalid_fixation_points import\
    filter_fixations_to_images
from preprocessing.map.map_internal_time_to_unix import\
    map_internal_time_to_unix
from preprocessing.parser.parse_asc_file import parse_asc_file_for_EFIX_events

from preprocessing.parser.parse_event_file import parse_event_file
from preprocessing.match.match_experiment_events_to_experiment_group import\
    match_experiment_events_to_experiment_group
from preprocessing.match.match_eye_fixations_timestamps_to_experiment_images\
    import match_eye_fixations_timestamps_to_experiment_images
from preprocessing.match.match_eye_fixations_to_element_labels import\
    match_eye_fixations_to_element_labels
from preprocessing.match.match_subject_answers import\
    match_subject_answers_to_question_views
from preprocessing.filter.filter_fixations_to_words import\
    filter_fixatios_to_words
from preprocessing.parser.parse_text_bounding_boxes import\
    parse_text_bounding_box_file


def get_user_data_from_submit(s: Dict) -> List[Any]:
    """Gets the perosonal data from the submit.

    :param s: The submit event as dict.
    :type s: Dict
    :return: List for later construction of a DF.
    :rtype: List[Any]
    """
    return [s['workerId'],
            s['results']['outputs']['surveyData']['gender'],
            s['results']['outputs']['surveyData']['ageGroup'],
            s['results']['outputs']['surveyData']['ethnicity'],
            s['results']['outputs']['surveyData']['education'],
            s['results']['outputs']['surveyData']['vizExperience']]


def preprocess_subject(
    group: int,
    df_element_labels: pd.DataFrame,
    asc_file: str,
    event_file: str,
    experiment_group_json: str,
    experiment_desc_html_json: str,
    dataset_dir: str,
    timestamp_unix: int,
    time_internal_unix: int,
    time_internal_main: int,
    time_main_content: int,
    good_eye: str,
    baseline: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full end to end processing of a single subject.

    :param group: Group number.
    :type group: int
    :param df_element_labels: Dir to element labels.
    :type df_element_labels: pd.DataFrame
    :param asc_file: ASC file of the subject.
    :type asc_file: str
    :param event_file: The event file of the subject.
    :type event_file: str
    :param experiment_group_json: The experiment description of this group.
    :type experiment_group_json: str
    :param images_dir: Dir to all images.
    :type images_dir: str
    :param timestamp_unix: Timestamp in unix debugout file.
    :type timestamp_unix: int
    :param time_internal_unix: Internal log file timestamp that corresponds
        to unix time.
    :type time_internal_unix: int
    :param time_internal_main: Internal log file timestamp that corresponds
        to the ASC time.
    :type time_internal_main: int
    :param time_main_content: Timestamp from ASC fixations.
    :type time_main_content: int
    :param good_eye: Wether to only keep RIGHT/LEFT.
    :type good_eye: str
    :return: Returns 4 DF for fixations, basic subject data, questions the
        subject saw etc.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    df_fixations = parse_asc_file_for_EFIX_events(asc_file, good_eye)
    df_fixations = map_internal_time_to_unix(
        df_fixations,
        (timestamp_unix, time_internal_unix,
         time_main_content, time_internal_main))

    df_events, submits = parse_event_file(event_file)

    json_file = open(experiment_group_json)
    experiment_group = json.load(json_file)
    json_file.close()

    json_file = open(experiment_desc_html_json)
    experiment_group_html = json.load(json_file)
    json_file.close()

    df_image_times, df_question_views, _\
        = match_experiment_events_to_experiment_group(
            df_events, experiment_group, baseline)

    df_question_views = match_subject_answers_to_question_views(
        df_question_views, submits[0][1])

    df_images_with_fixations\
        = match_eye_fixations_timestamps_to_experiment_images(
            df_fixations, df_image_times)

    # df_crosses_with_fixations\
    #     = match_eye_fixations_timestamps_to_experiment_images(
    #         df_fixations, df_cross_views)

    df_question_images_with_fixations =\
        df_images_with_fixations[
            df_images_with_fixations['kind'] == 'question']
    if baseline:
        df_baseline_text_boxes = parse_text_bounding_box_file(
            f"dataset/TextBB_Baseline/TextBB_G{group}.txt")
        df_words_with_fixations = \
            filter_fixatios_to_words(
                df_question_images_with_fixations,
                df_baseline_text_boxes,
                experiment_group_html)
    else:
        df_baseline_text_boxes = parse_text_bounding_box_file(
            f"dataset/TextBB/TextBB_G{group}.txt")
        df_words_with_fixations = \
            filter_fixatios_to_words(
                df_question_images_with_fixations,
                df_baseline_text_boxes,
                experiment_group_html)

    df_images_with_fixations_filtered = filter_fixations_to_images(
        group, df_images_with_fixations, dataset_dir, baseline)

    df_eye_fixations_to_elements\
        = match_eye_fixations_to_element_labels(
            df_images_with_fixations_filtered, df_element_labels)
    user_data = get_user_data_from_submit(submits[0][1])

    return df_eye_fixations_to_elements, df_question_views,\
        user_data, df_words_with_fixations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asc_file", type=str, default=None)
    parser.add_argument("--event_file", type=str, default=None)
    parser.add_argument("--experiment_group_json", type=str, default=None)
    parser.add_argument("--timestamp_unix", type=int, default=None)
    parser.add_argument("--timestamp_internal", type=int, default=None)
    args = vars(parser.parse_args())

    preprocess_subject(args['asc_file'], args['event_file'],
                       args['experiment_group_json'],
                       args['timestamp_unix'],
                       args['timestamp_internal'])
