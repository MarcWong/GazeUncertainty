from typing import Tuple

import pandas as pd


TAlignement = Tuple[int, int, int, int]


def align_to_unix(
    internal_timestamp: int,
    alignement_timestamps: TAlignement
) -> int:
    """Aligns a timestamp given an example of how the timestamp is related to
    the usual unix time.

    :param internal_timestamp: Timestamp to align.
    :type internal_timestamp: int
    :param alignement_timestamps: Tuple that serves as evidence.
    :type alignement_timestamps: TAlignement
    :return: Aligned timestamp.
    :rtype: int
    """
    timestamp_unix = alignement_timestamps[0]
    time_internal_unix = alignement_timestamps[1]
    time_main_content = alignement_timestamps[2]
    time_internal_main = alignement_timestamps[3]

    correspondence = time_main_content + time_internal_unix - time_internal_main

    return int(internal_timestamp + (timestamp_unix - correspondence))


def map_internal_time_to_unix(
    df_fixations: pd.DataFrame,
    alignement_timestamps: TAlignement
) -> pd.DataFrame:
    """Maps internal timestamps to unix ones given one pair of evidence as
    alignement initialization.

    :param df_fixations: DF containing a <start_time> and <end_time> column
        to fix.
    :type df_fixations: pd.DataFrame
    :param alignement_timestamps: Tuple(unix_time, corresponding_internal_time)
        that serves as evidence.
    :type alignement_timestamps: TAlignement
    :return: Aligned dataframe.
    :rtype: pd.DataFrame
    """
    df_fixations['start_time'] = df_fixations['start_time'].apply(
        lambda t: align_to_unix(t, alignement_timestamps)
    )
    df_fixations['end_time'] = df_fixations['end_time'].apply(
        lambda t: align_to_unix(t, alignement_timestamps)
    )
    return df_fixations
