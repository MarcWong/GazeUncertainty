import argparse
import pandas as pd


def compute_refixation_per_img(
    df_eye_fixation_with_elem_labels: pd.DataFrame
) -> pd.DataFrame:
    """Computes the number of refixations, i.e. the number of distinct looks at
    an element -1 (for the initial look).

    df_eye_fixation_with_elem_labels (image_name,kind,image_flag,stage,
                                      start_time,end_time,EVENT,eye,fix_start_time,
                                      fix_end_time,dur,axp,ayp,aps,file,id,desc,
                                      polygon,contains)

    :param df_eye_fixation_with_elem_labels: The eye fixations with elem
        labels.
    :type df_eye_fixation_with_elem_labels: pd.DataFrame
    :return: New dataframe containing. (refixation_count, image_name, file, id)
    :rtype: pd.DataFrame
    """
    last_file = None
    last_id = None
    fixations = {
        'refixation_count': [],
        'image_name': [],
        'file': [],
        'id': [],
        'desc': []
    }

    idx_id_exists_idx = {}

    for _, image_name, *_, elem_id, desc, file, _, contains, _\
            in df_eye_fixation_with_elem_labels.itertuples():
        if elem_id is None:
            last_file = file
            last_id = elem_id
            continue
        if contains and last_file != file and last_id != elem_id:
            idx_id = file + str(elem_id)

            try:
                idx = idx_id_exists_idx[idx_id]
                fixations['refixation_count'][idx] += 1
            except KeyError:
                idx_id_exists_idx[idx_id] = len(fixations['id'])
                fixations['refixation_count'].append(0)
                fixations['image_name'].append(image_name)
                fixations['desc'].append(desc)
                fixations['file'].append(file)
                fixations['id'].append(elem_id)
        last_file = file
        last_id = elem_id
    return pd.DataFrame.from_records(fixations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye_fixation_with_elem_labels",
                        type=str, default=None)
    args = vars(parser.parse_args())
    if args['eye_fixation_with_elem_labels']:
        path = args['eye_fixation_with_elem_labels']
        df = pd.read_csv(path)
        print(compute_refixation_per_img(df))
