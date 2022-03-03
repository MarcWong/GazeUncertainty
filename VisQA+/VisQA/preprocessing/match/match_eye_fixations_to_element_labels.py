import os
import argparse
import pandas as pd
from glob import glob

from scipy.spatial.distance import euclidean
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ast import literal_eval


def label_imagename(row: pd.Series, df_fixations: pd.DataFrame):
    """Finds the correct image file name with ending to
    a name without the ending.

    :param row: The row with the filename.
    :type row: pd.Series
    :param df_fixations: The fixations DF that includes all image file names.
    :type df_fixations: pd.DataFrame
    :return: The altered row including the image file name.
    :rtype: pd.Row
    """
    img_name_row = df_fixations[
        (df_fixations['image_name'] == row['file'] + ".jpg") |
        (df_fixations['image_name'] == row['file'] + ".png")]

    if img_name_row.shape[0] != 0:
        row['image_name'] = img_name_row['image_name'].values[0]
    else:
        row['image_name'] = ''
        ValueError('More than one or zeros row detected.')
    return row


def compute_is_in_polygon(row: pd.Series) -> pd.Series:
    """Checks wether the fixation point <axp>, <ayp> is in the polygon of the row.
    BUT: It turns out that polygon only sometimes resembles a polygon.
    There can also be: Arbitrary complicated lines, circles, single data points etc.

    :param row: The row to compute contains on.
    :type row: pd.Series
    :return: The row with ['contains'] added.
    :rtype: pd.Series
    """
    point = Point(int(row['axp']), int(row['ayp']))
    points = row['polygon']
    if len(points) == 2 and '(circle)' in row['desc']:
        r = euclidean(points[0], points[1])
        row['contains'] = (point.x - points[0][0])**2 + \
            (point.y - points[0][1])**2 < r**2
    elif len(points) == 2 or '(line)' in row['desc']:
        for seg in range(1, len(row['polygon'])):
            dxc = point.x - points[seg-1][0]
            dyc = point.y - points[seg-1][1]
            dxl = points[seg][0] - points[seg-1][0]
            dyl = points[seg][1] - points[seg-1][1]
            cross = dxc * dyl - dyc * dxl
            if cross == 0:
                row['contains'] = True
        row['contains'] = False
    elif len(points) == 1 and '(point)' in row['desc']:
        row['contains'] = point.x == points[0][0] and point.y == points[0][1]
    elif len(points) < 3:
        row['contains'] = False  # Default handler.
    else:
        polygon = Polygon(points)
        row['contains'] = polygon.contains(point)
    return row


def reduce_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the df and reduces the rows to either
    a single row representing no element being fixated or
    1 or 2 eyes fixating an element in 1 or 2 rows.

    :param df: Dataframe from group by the fix time stamps.
    :type df: pd.DataFrame
    :return: DataFrame with the fixation rows.
    :rtype: pd.DataFrame
    """
    contains = df[df['contains'] == True]
    if contains.shape[0] >= 1:
        return contains
    else:
        sr = df.iloc[0].to_dict()
        sr['contains'] = False
        sr['polygon'] = None
        sr['desc'] = None
        sr['id'] = None
        return pd.DataFrame.from_records([sr])


def match_eye_fixations_to_element_labels(
    df_fixations: pd.DataFrame,
    df_element_labels: pd.DataFrame
) -> pd.DataFrame:
    """Matches an eye fixation dataframe against the relevant element labels
    sucht that every fixation is associated with an corresponding element label.

    df_fixations (image_name, kind, image_flag, stage, start_time, end_time,
                  EVENT, eye, fix_start_time, fix_end_time, dur, axp, ayp, aps)

    df_element_labels (id, desc, file, polygon)

    :param df_fixations: Fixations DF (see doc).
    :type df_fixations: pd.DataFrame
    :param df_element_labels: DF containing element labels.
    :type df_element_labels: pd.DataFrame
    :return: Combined dataframe.
    :rtype: pd.DataFrame
    """
    df_element_labels = df_element_labels.apply(
        lambda row: label_imagename(row, df_fixations), axis=1)
    df = pd.merge(df_fixations, df_element_labels, on='image_name')
    df = df.apply(lambda row: compute_is_in_polygon(row), axis=1)
    df = df.groupby(['fix_start_time', 'fix_end_time'],
                    as_index=False).apply(reduce_rows)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, default=None)
    args = vars(parser.parse_args())

    base_path = '/netpool/homes/wangyo/Dataset/VisQA/eyetracking/csv_files/fixationsByVis/bar'
    subjects_path = glob(os.path.join(
    f"{base_path}/*", "enc", "*"))

    subjects_path = [os.path.abspath(path) for path in subjects_path]
    print(subjects_path)

    for path in subjects_path:
        visname = path.split('/')[-3]
        #print(visname)
        eyes = pd.read_csv(path)
        elem = pd.read_csv(args['element_labels_dir'] + f'/{visname}')
        #print(fixations)

        #TODO: fix here
        df = match_eye_fixations_to_element_labels(eyes, elem)
        print(df)