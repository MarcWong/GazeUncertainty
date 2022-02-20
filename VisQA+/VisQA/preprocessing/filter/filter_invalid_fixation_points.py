from typing import Tuple

import ast
import argparse
import pandas as pd

from PIL import Image


Tx, Ty, Tw, Th, min, max = int, int, int, int, int, int
TLimit = Tuple[min, max]
TImageSize = Tuple[Tw, Th]
TCoord = Tuple[Tx, Ty]

MAX_VISQA_IMAGE_WIDTH = 1066.666
MAX_VISQA_IMAGE_HEIGHT = 800
MAX_VISQA_IMAGE_WIDTH_GROUP_4 = 1600
MAX_VISQA_IMAGE_HEIGHT_GROUP_4 = 770
DISPLACEMENT_TOP = 123.5
DISPLACEMENT_LEFT = 1920//2 - 1066.666//2


def filter_fixations_by_bounding_box(
    df: pd.DataFrame,
    image_name: str,
    x_limits: TLimit,
    y_limits: TLimit,
    scale_factor: float = 1.,
) -> pd.DataFrame:
    """
    Filters eye fixations DF that do not lie in a predefined box, i.e.
    the image we are interested in.

    MUST CONTAIN:
    ... image_name axp ayp ...

    Group 4 needs special handeling since their images were scaled differently.

    :param df: DF to filter with the form as above.
    :param image_name: The name of the image.
    :param x_limits: Tuple (min x, max x) both inclusive.
    :param y_limits: Tuple (min y, max y) both inclusive.
    :param scale_factor: If the image was scaled the points need to be
        descaled in order to fit the image.
    :returns: Filtered dataframe.
    """
    df = df[(df['image_name'] == image_name)
            & (x_limits[0] <= df['axp'])
            & (df['axp'] <= x_limits[1])
            & (y_limits[0] <= df['ayp'])
            & (df['ayp'] <= y_limits[1])]
    if df.empty:
        return df
    df['axp'] = df.apply(
        lambda row: int((row['axp'] - x_limits[0]) /
                        scale_factor) if row['image_name'] == image_name else row['axp'],
        axis=1
    )
    df['ayp'] = df.apply(
        lambda row: int((row['ayp'] - y_limits[0]) /
                        scale_factor) if row['image_name'] == image_name else row['ayp'],
        axis=1
    )
    return df


def filter_fixatios_to_image_box(
        df: pd.DataFrame,
        image_name: str,
        image_size: TImageSize,
        max_visqa_image_width: int = 1066.666,
        max_visqa_image_height: int = 800,
        displacement_top: float = 123.5,
        displacement_left: float = 1920//2-1066.666//2) -> pd.DataFrame:
    """Takes a list of fixations from a 1920x1080 screen and fits it to an image
    that is displayes in a max_visqa_image_width x max_visqa_image_height
    bounding box offset from the top by displacement_top and from the left by
    displacement_left.

    :param df: Fixations data set.
    :type df: pd.DataFrame
    :param image_name: The name of the image.
    :type image_name: str
    :param image_size: The size of the original image (not scaled to bounding box).
    :type image_size: TImageSize
    :param max_visqa_image_width: Max bounding box wdith, defaults to 1066.666
    :type max_visqa_image_width: int, optional
    :param max_visqa_image_height: Max bounding box height, defaults to 800
    :type max_visqa_image_height: int, optional
    :param displacement_top: Displacement of the box from the top, defaults to 123.5
    :type displacement_top: float, optional
    :param displacement_left: Displacement of the box from the left, defaults to 1920//2-1066.666//2
    :type displacement_left: float, optional
    :return: Filtered df containing fixations that only look at the image
        (instead of the whole screen).
    :rtype: pd.DataFrame
    """
    image_center: TCoord = (displacement_left + (max_visqa_image_width//2),
                            displacement_top + (max_visqa_image_height//2))
    w, h = image_size
    if max_visqa_image_height / h < max_visqa_image_width / w:
        scale_factor = max_visqa_image_height / h
    else:
        scale_factor = max_visqa_image_width / w
    w, h = scale_factor * w, scale_factor * h

    return filter_fixations_by_bounding_box(
        df, image_name,
        (image_center[0]-w//2,
         image_center[0]+w//2),
        (image_center[1]-h//2,
         image_center[1]+h//2),
        scale_factor=scale_factor)


def filter_fixations_to_images(
    group: int, df: pd.DataFrame, dataset_dir: str, baseline: bool
) -> pd.DataFrame:
    """Takes a list of fixations from a 1920x1080 screen and fits it to an image
    that is displayes in a max_visqa_image_width x max_visqa_image_height
    bounding box offset from the top by displacement_top and from the left by
    displacement_left.

    :param group: Group number.
    :type group: int
    :param df: Fixations data set.
    :type df: pd.DataFrame
    :param image_dir: Path to all experiment images
    :type image_dir: str
    :return: Filtered df containing fixations that only look at the image 
        (instead of the whole screen).
    :rtype: pd.DataFrame
    """
    final = []
    df_enc = df[(df['stage'] != 'recall') &
                (df['image_flag'] != 'blur')]
    df_blurred = df[(df['stage'] == 'recall') &
                    (df['image_flag'] == 'blur')]

    images = list(df_enc['image_name'].unique())
    for image in images:
        im = Image.open(f"{dataset_dir}/merged/src/{image}")
        width, height = im.size
        im.close()
        df_in = df_enc[df_enc['image_name'] == image]
        if baseline:
            if group == 10:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1035.22,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1035.22//2)
            elif group == 1 or group == 5 or group == 6 or group == 8:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1066.666,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1066.666//2)
            elif group == 2:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1169,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1170//2)
            elif group == 3:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1069.25,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1069.25//2)
            elif group == 4:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1600,
                    max_visqa_image_height=774.98,
                    displacement_top=125,
                    displacement_left=1920//2-1600//2)
            elif group == 7:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1023.28,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1023.28//2)
            elif group == 9:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1142.67,
                    max_visqa_image_height=800,
                    displacement_top=125,
                    displacement_left=1920//2-1142.67//2)
            else:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1600,
                    max_visqa_image_height=770,
                    displacement_top=125,
                    displacement_left=1920//2-1600//2)
        elif group != 4:
            df_r = filter_fixatios_to_image_box(
                df_in, image, (width, height))
        else:
            df_r = filter_fixatios_to_image_box(
                df_in, image, (width, height),
                max_visqa_image_width=1600,
                max_visqa_image_height=770,
                displacement_top=123.5,
                displacement_left=1920//2-1600//2)
        final.append(df_r)

    images = list(df_blurred['image_name'].unique())
    for image in images:
        im = Image.open(f"{dataset_dir}/merged/src/{image}")
        width, height = im.size
        im.close()
        for question_id in range(1, 6):
            df_in = df_blurred[(df_blurred['image_name'] == image)
                               & (df_blurred['question_id'] == question_id)]
            if baseline:
                if group == 10:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1036,
                        max_visqa_image_height=800,
                        displacement_top=123.5,
                        displacement_left=1920//2-1036//2)
                elif group == 1 or group == 5 or group == 6 or group == 8:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1066.666,
                        max_visqa_image_height=798,
                        displacement_top=123.5,
                        displacement_left=1920//2-1066.666//2)
                elif group == 2:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1170,
                        max_visqa_image_height=789,
                        displacement_top=123.5,
                        displacement_left=1920//2-1170//2)
                elif group == 3:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1070,
                        max_visqa_image_height=798,
                        displacement_top=123.5,
                        displacement_left=1920//2-1070//2)
                elif group == 4:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1600,
                        max_visqa_image_height=770,
                        displacement_top=123.5,
                        displacement_left=1920//2-1600//2)
                elif group == 7:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1023,
                        max_visqa_image_height=798,
                        displacement_top=123.5,
                        displacement_left=1920//2-1023//2)
                elif group == 9:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1143,
                        max_visqa_image_height=798,
                        displacement_top=123.5,
                        displacement_left=1920//2-1143//2)
                else:
                    df_r = filter_fixatios_to_image_box(
                        df_in, image, (width, height),
                        max_visqa_image_width=1600,
                        max_visqa_image_height=770,
                        displacement_top=123.5,
                        displacement_left=1920//2-1600//2)
            elif group != 4:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1066.666//2,
                    max_visqa_image_height=800//2,
                    displacement_top=123.5,
                    displacement_left=1920//2-1066.666//4)
            else:
                df_r = filter_fixatios_to_image_box(
                    df_in, image, (width, height),
                    max_visqa_image_width=1600//2,
                    max_visqa_image_height=770//2,
                    displacement_top=123.5,
                    displacement_left=1920//2-1600//4)
            final.append(df_r)
    return pd.concat(final)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixation_asc_csv_file", type=str, default=None)
    parser.add_argument("--x_lims", type=ast.literal_eval, default=(0, 1920))
    parser.add_argument("--y_lims", type=ast.literal_eval, default=(0, 1080))
    args = vars(parser.parse_args())

    if args['asc_file'] is not None:
        print(filter_fixatios_to_image_box(
            args['fixation_asc_csv_file'], args['x_lims'], args['y_lims']))
