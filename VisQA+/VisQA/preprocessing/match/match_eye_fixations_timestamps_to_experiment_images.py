import argparse

import pandas as pd


def match_eye_fixations_timestamps_to_experiment_images(
    df_fixations: pd.DataFrame,
    df_image_times: pd.DataFrame
) -> pd.DataFrame:
    """Joins two Dataframes given image presention times
    and the eye fixations.

    :param df_fixations: DF describing eye fixations durations and places
        filtered to the image. Columns:
            EFIX <eye> <start_time> <end_time> <dur> <axp> <ayp> <aps>
    :type df_fixations: pd.DataFrame
    :param df_image_times: DF describing which images are looked at when.
        Columns: image_name, stage, start_time, end_time
    :type df_image_times: pd.DataFrame
    :return: Joined Dataframe
    :rtype: pd.DataFrame
    """
    df_images_with_fixations = []

    for img_time in df_image_times.itertuples():
        hits = df_fixations[(df_fixations['end_time'] > img_time.start_time)
                            & (df_fixations['start_time'] < img_time.end_time)]
        for hit in hits.itertuples():
            df_images_with_fixations.append(
                (img_time.image_name, img_time.kind, img_time.image_flag,
                 img_time.stage, img_time.start_time,
                 img_time.end_time, img_time.question_id, hit.event, hit.eye,
                 hit.start_time, hit.end_time, hit.dur, hit.axp, hit.ayp,
                 hit.aps, hit.gaze)
            )

    df_images_with_fixations = pd.DataFrame.from_records(
        df_images_with_fixations,
        columns=['image_name', 'kind', 'image_flag', 'stage', 'start_time',
                 'end_time', 'question_id', 'EVENT', 'eye', 'fix_start_time',
                 'fix_end_time', 'dur', 'axp', 'ayp', 'aps', 'gaze'])
    return df_images_with_fixations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye_fixations_csv", type=str, default=None)
    parser.add_argument("--image_timestamps_csv", type=str, default=None)
    args = vars(parser.parse_args())

    if (args['eye_fixations_csv'] is not None
            and args['image_timestamps_csv'] is not None):
        match_eye_fixations_timestamps_to_experiment_images(
            args['eye_fixations_csv'], args['image_timestamps_csv'])
