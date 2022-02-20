import pandas as pd

from analysis.compute_refixations.compute_refixations\
    import compute_refixation_per_img
# from analysis.compute_refixations.compute_fixation_densities\
#     import compute_refixation_densities


def analyse_subject(
    df_eye_fixation_with_elem_labels: pd.DataFrame
) -> pd.DataFrame:
    """Performs analysis on the subject.

    :param df_eye_fixation_with_elem_labels: Pandas DataFrame from
        eye_fixation_with_elem_labels
    :type df_eye_fixation_with_elem_labels: pd.DataFrame
    :return: DF computes refixations.
    :rtype: pd.DataFrame
    """
    return compute_refixation_per_img(df_eye_fixation_with_elem_labels)
