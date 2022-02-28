import numpy as np
import os.path
import argparse
import logging
import json

from glob import glob


def flipping_candidate_score(densities):
    """
    Flipping candidates score has the following interpretation:
    ~> 0: The density distribution is peaked, i.e. the fixation mostly covers just a single AOI.
    ~> 1: The density distribution is close to uniform, i.e. the fixation covers at least two AOI to a very similar extent.

    NOTE: Is there off-the-shelf solution for this? There might be better / more elegant way to compute this.
    """
    def deviation_from_uniform(densities, N):
        uniform = 1 / N
        return sum([abs(d - uniform) for _, d in densities])

    sorted_densities = sorted(densities, reverse=True, key=lambda x: x[1])
    N = len(densities)
    max_score = 0

    # Flipping candidates switch between at least two AOIs
    for n in range(2, N+1):
        deviation = deviation_from_uniform(sorted_densities[:n], n)
        score = 1. - ((2 / n) * deviation)
        max_score = max(score, max_score)
    return max_score


def find_flipping_candidates(fixation_densities):
    """
    Perform KDE analysis steps to find flipping candidates.
    Output can be verified with show_density_overlay=True
    """
    flipping_threshold = 0.5
    flipping_candidates = []

    for densities in fixation_densities:
        sum_densities = sum([d for _, d in densities])
        if sum_densities < 1.:
            densities.append(('#', 1.-sum_densities))
        assert sum_densities <= 1+1e-1, f"Element labels are overlapping. Densities add up to {sum_densities:.3f}"

        # Step 6: check for which segments the distribution overlays at least two AOIs to a very similar extent (the flipping candidates)
        if len(densities) > 1:
            score = flipping_candidate_score(densities)
            logging.info(f'  Flipping candidate score = {score:.3f}: {densities}')
            if score >= flipping_threshold:
                flipping_candidates.append((densities))
    return flipping_candidates


def flipping_candidate_rate_of_vis(densities_dir):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    flipping_candidate_rate = []
    for path in glob(densities_dir, '*.json'):
        with open(path, 'r') as f:
            densities = json.load(f)

            flipping_candidates = find_flipping_candidates(densities)
            flipping_candidate_rate.append(len(flipping_candidates) / len(densities))
            logging.info(f'\nNumber of fixations being flipping candidates: {len(flipping_candidates)}/{len(densities)}\n')
    return flipping_candidate_rate


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    logging.basicConfig(filename='flipping_rate_analysis.log', filemode='w', level=logging.INFO)
    type2rate = {vt: [] for vt in vis_types}

    for vis_type in args['vis_types']:
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            #Flipping candidate ratios of all recordings associated to vis
            fc_rate = flipping_candidate_rate_of_vis(vis_densities)
            if len(fc_rate) > 0:
                avg_fc_rate = np.mean(fc_rate)
                type2rate[vis_type].append(avg_fc_rate)
                logging.info(f'\nAverage flipping candidate ratio: {avg_fc_rate:.5f}\n')
            else:
                logging.info('\nNo flipping candidates!\n')
        np.save(f'{vis_type}_FC_rates.npy', type2rate[vis_type])
