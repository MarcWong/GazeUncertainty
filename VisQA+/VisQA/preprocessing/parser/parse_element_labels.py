import os
import argparse
import pandas as pd

from glob import glob


def parse_element_label(element_label_path: str) -> pd.DataFrame:
    """Takes a element label path containing a list of bounding boxes
    and writes it into a dataframe.

    :param element_label_path: The file to the element label file.
    :type element_label_path: str
    :return: Parsed df. (id, description, x, y, file)
    :rtype: pd.DataFrame
    """
    file_name = os.path.basename(element_label_path)
    df = pd.read_csv(element_label_path, names=[
                     'id', 'description', 'x', 'y'])
    df['file'] = file_name
    return df


def combine_rows(df_element_labels: pd.DataFrame) -> pd.DataFrame:
    """Combines the rows by id by combining the x,y coordinates to 
    a list that represents a polygon.

    df_element_labels (id, description, x, y, file)

    :param df_element_labels: DF containing the element labels.
    :type df_element_labels: pd.DataFrame
    :return: New sequashed df (id, desc, file, polygon)
    :rtype: pd.DataFrame
    """
    df = []
    element_ids = df_element_labels['id'].unique()
    for id in element_ids:
        rows = df_element_labels[df_element_labels['id'] == id]
        f = rows.values[0]
        df.append(
            {"id": f[0], "desc": f[1], "file": f[-1], 'polygon': []})
        for _, id, _, x, y, _ in rows.itertuples():
            df[-1]['polygon'].append((x, y))
    return pd.DataFrame.from_records(df)


def parse_element_labels(element_label_dir: str) -> pd.DataFrame:
    """Takes all paths in the element label directory and returns a
    single DF containing all boxes described through the files.

    :param df_element_label_dir: Directory to parse; Must only contain
        bounding boxes file.
    :type df_element_label_dir: str
    :return: Complete DF with element labels and boxes.
    :rtype: pd.DataFrame
    """
    element_labels_path = glob(os.path.join(element_label_dir, "*"))
    element_labels_path = [
        os.path.abspath(path) for path in element_labels_path]

    element_labels = [
        parse_element_label(path) for path in element_labels_path]
    element_labels = [
        combine_rows(df) for df in element_labels]
    return pd.concat(element_labels, ignore_index=True, sort=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_label_dir", type=str, default=None)
    args = vars(parser.parse_args())

    if args['element_label_dir'] is not None:
        df = parse_element_labels(args['element_label_dir'])
        print(df.head(20))
        df.to_csv(args['element_label_dir'] + '/element_labels.csv')
