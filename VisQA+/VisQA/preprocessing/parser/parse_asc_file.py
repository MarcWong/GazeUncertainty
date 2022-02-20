import re
import argparse
import pandas as pd


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
        lines = list(filter(lambda x: not re.match(
            r'^\s*$|\*\*.*', x), lines))
        if good_eye is not None:
            eye = good_eye[0]
            lines = list(filter(lambda x: x.startswith(f"EFIX {eye}"), lines))
        else:
            lines = list(filter(lambda x: x.startswith("EFIX"), lines))
        lines = list(map(lambda x: x.split(), lines))
        # EFIX <eye> <start_time> <end_time> <dur> <axp> <ayp> <aps>
        lines = [[l[0], l[1], int(l[2]), int(l[3]),
                  float(l[4]), float(l[5]), float(l[6]),
                  int(l[7])] for l in lines]
        df = pd.DataFrame.from_records(
            lines, columns=['event', 'eye', 'start_time',
                            'end_time', 'dur', 'axp', 'ayp', 'aps'])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asc_file", type=str, default=None)
    args = vars(parser.parse_args())

    if args['asc_file'] is not None:
        print(parse_asc_file_for_EFIX_events(args['asc_file']))
