import subprocess
import argparse

from glob import glob


def edf_to_asc(edf_file: str, out_dir: str = None) -> str:
    """
    Calls the edf2asc command and saves it.

    :param edf_file: Path to edf file.
    :param out_dir: Where to place the asc file.
    :returns: Path to asc file.
    """
    assert(edf_file[-4:] == ".edf")
    if out_dir is None:
        out_file = f'{edf_file[:-4]}.asc'
    else:
        out_file = f'{out_dir}/{edf_file.split("/")[-1][:-4]}.asc'
    if out_dir is not None:
        subprocess.run(["edf2asc", edf_file, out_file, "-sg" , "-y"])
    else:
        subprocess.run(["edf2asc", edf_file, "-sg", "-y"])
    return out_file


if __name__ == '__main__':
    print("WARNING: This only calls eyelinks edf2asc command with paramters appropiate to this project.")
    print("WARNING: Requires edf2asc")
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_file", type=str, default=None)
    parser.add_argument("--edf_in_dir", type=str, default=None)
    parser.add_argument("--edf_glob_pattern", type=str, default="*/*.edf")
    parser.add_argument("--asc_out_dir", type=str, default=None)
    args = vars(parser.parse_args())

    if args['edf_file'] is not None:
        print(edf_to_asc(args['edf_file'], args['asc_out_dir']))
    elif args['edf_in_dir'] is not None:
        for path in glob(f"{args['edf_in_dir']}/{args['edf_glob_pattern']}"):
            edf_to_asc(path, args['asc_out_dir'])
