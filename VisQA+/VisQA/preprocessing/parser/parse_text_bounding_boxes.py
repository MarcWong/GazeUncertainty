import argparse
import pandas as pd

from io import StringIO


def parse_text_bounding_box_file(path: str) -> pd.DataFrame:
    """Given a path to a bounding box file for the text in a group, 
    return a DataFrame containing the bounding boxes.

    :param path: Path to groups bounding boxes.
    :type path: str
    :return: DF containing all DF of this group describing the text.
    :rtype: pd.DataFrame
    """
    with open(path, 'r') as f:
        content = f.read()
        lines = content.splitlines()
        lines = list(
            filter(lambda x: not "" and not x.startswith("----"), lines))
        data = StringIO("\n".join(lines))
    return pd.read_csv(data, sep=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_bounding_boxes", type=str, default=None)
    args = vars(parser.parse_args())

    if args['text_bounding_boxes'] is not None:
        print(parse_text_bounding_box_file(args['text_bounding_boxes']))
