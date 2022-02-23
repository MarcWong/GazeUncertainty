import re
import argparse
import pandas as pd
from itertools import chain


def parse_asc_file_for_EFIX_events(
    asc_file: str,
    good_eye: str = None
) -> pd.DataFrame:
    """
    EFIX Line parsing; Formally:
    EFIX <eye> <start_time> <end_time> <dur> <axp> <ayp> <aps>
    Example:
    EFIX L   8680694	8681162	468	  983.9	  545.1	   3644

    :param asc_file: File path to the asc file.
    :type asc_file: str
    :param good_eye: Wether the analysis should only keep one good eye, 
        defaults to None
    :type good_eye: str, optional
    :returns: Dataframe containing the EFIX event as df with headers as above.
    :rtype: pd.DataFrame
    """
    with open(asc_file, 'r') as file:
        content = file.read()
        lines = content.splitlines()

        fix_lines = list(filter(lambda x: not re.match(
            r'^\s*$|\*\*.*', x), lines))
        if good_eye is not None:
            eye = good_eye[0]
            fix_lines = list(filter(lambda x: x.startswith(f"EFIX {eye}"), fix_lines))
            sfix_line_num = list(filter(lambda row: lines[row].startswith(f"SFIX {eye}"), range(len(lines))))
            efix_line_num = list(filter(lambda row: lines[row].startswith(f"EFIX {eye}"), range(len(lines))))
        else:
            fix_lines = list(filter(lambda x: x.startswith("EFIX"), fix_lines))
            sfix_line_num = list(filter(lambda row: lines[row].startswith("SFIX"), range(len(lines))))
            efix_line_num = list(filter(lambda row: lines[row].startswith("EFIX"), range(len(lines))))

        gaze_lines = []
        for fix_start, fix_end in zip(sfix_line_num, efix_line_num):
            within_fix_lines = lines[fix_start+1: fix_end]
            within_fix_lines = list(map(lambda x: x.split(), within_fix_lines))
            # Each gaze sample has nine entries in our case and timestamp (first column) is a number.
            within_fix_gaze_lines = list(filter(lambda x: len(x) == 9 and x[0].isnumeric(), within_fix_lines))
            flattened = chain.from_iterable(within_fix_gaze_lines)
            flattened = '\t'.join(flattened)
            gaze_lines.append(flattened)

        fix_lines = list(map(lambda x: x.split(), fix_lines))

        assert len(fix_lines) == len(gaze_lines), "Number of fixation lines must equal the number of gaze lines"

        # EFIX <eye> <start_time> <end_time> <dur> <axp> <ayp> <aps> <gaze>
        fix_lines = [[l[0], l[1], int(l[2]), int(l[3]),
                  float(l[4]), float(l[5]), float(l[6]),
                  int(l[7]), g] for l, g in zip(fix_lines, gaze_lines)]
        df = pd.DataFrame.from_records(
            fix_lines, columns=['event', 'eye', 'start_time',
                            'end_time', 'dur', 'axp', 'ayp', 'aps', 'gaze'])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asc_file", type=str, default=None)
    args = vars(parser.parse_args())

    if args['asc_file'] is not None:
        print(parse_asc_file_for_EFIX_events(args['asc_file']))
