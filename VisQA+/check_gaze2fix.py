import os.path
import numpy as np
import pandas as pd
import argparse

from glob import glob
from os.path import basename

def parse_gaze_samples(xp_str, yp_str):
    xps = xp_str[1:-1].split(';')
    yps = yp_str[1:-1].split(';')
    xps = np.array(xps).astype(float)
    yps = np.array(yps).astype(float)
    return np.stack((xps, yps), axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--max-pixel-error", type=int, default=3)
    args = vars(parser.parse_args())

    for img_dir in glob(os.path.join(args['dataset_dir'], 'eyetracking/csv_files/fixationsByVis/*/*')):
        img_src = basename(img_dir)
        for fixation_path in glob(os.path.join(img_dir, 'enc/*.csv')):
            viewer_id = basename(fixation_path)[:-4]
            fixation = pd.read_csv(fixation_path, delimiter=',')
            for index, row in fixation.iterrows():
                n, fx, fy, dur, gx, gy = row

                gaze_samples = parse_gaze_samples(gx, gy)
                avg_gx, avg_gy = np.mean(gaze_samples, axis=0).round()

                if abs(avg_gx - fx) > args['max_pixel_error'] and abs(avg_gy - fy) > args['max_pixel_error']:
                    print(f'{viewer_id}: error at fixation index={n:03}, fx={fx}, fy={fy}, dur={dur}: avg_gx: {avg_gx}; avg_gy: {avg_gy}')