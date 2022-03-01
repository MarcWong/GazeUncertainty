"""
Step 6: Calculates flipping candidate rate (fcr) of each visualization type
Requires precomputed densities for each visualization (Step 4-5), which can be obtained from 'kde_densities.py'
"""

import numpy as np
import os.path
import argparse
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob


def parse_densities(file):
    densities = json.load(file).values()
    for ds in densities:
        sum_densities = sum([d for _, d in ds])
        if sum_densities < 1.:
            ds.append(('#', 1.-sum_densities))
        #assert sum_densities <= 1+1e-1, f"Element labels are overlapping. Densities add up to {sum_densities:.3f}"
    return densities


def flipping_candidate_score_of_rank(densities, r):
    """
    Flipping candidates score has the following interpretation:
    ~> 0: The density distribution is peaked, i.e. the fixation mostly covers just a single AOI.
    ~> 1: The density distribution is close to uniform, i.e. the fixation covers at least two AOI to a very similar extent.

    NOTE: Is there off-the-shelf solution for this? There might be better / more elegant way to compute this.
    """
    def deviation_from_uniform(densities, N):
        uniform = 1 / N
        return sum([abs(d - uniform) for _, d in densities])

    N = len(densities)
    copy = list(densities)

    # Add dummy zeros when rank is larger than number density entries.
    if r > N:
        copy.extend([('0', 0)] * (r - N))

    copy.sort(reverse=True, key=lambda x: x[1])
    deviation = deviation_from_uniform(copy[:r], r)
    score = 1. - ((2 / r) * deviation)
    return score


def flipping_candidate_score(densities, ranks=(2, 3, 4)):
    """
    Flipping candidates score has the following interpretation:
    ~> 0: The density distribution is peaked, i.e. the fixation mostly covers just a single AOI.
    ~> 1: The density distribution is close to uniform, i.e. the fixation covers at least two AOI to a very similar extent.

    NOTE: Is there off-the-shelf solution for this? There might be better / more elegant way to compute this.
    """
    scores = [flipping_candidate_score_of_rank(densities, r) for r in ranks]
    return max(scores)


def find_flipping_candidates(fixation_densities, threshold):
    """
    Perform KDE analysis steps to find flipping candidates.
    Output can be verified with show_density_overlay=True
    """
    flipping_candidates = []
    for densities in fixation_densities:
        # Step 6: check for which segments the distribution overlays at least two AOIs to a very similar extent (the flipping candidates)
        score = flipping_candidate_score_of_rank(densities, r=10)
        logging.info(f'  Flipping candidate score = {score:.3f}: {densities}')
        if score >= threshold:
            flipping_candidates.append((densities))

    return flipping_candidates


def fcr_of_vis(densities_dir, flipping_threshold):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    flipping_candidate_rate = []
    for path in glob(os.path.join(densities_dir, '*.json')):
        with open(path, 'r') as f:
            densities = parse_densities(f)
            flipping_candidates = find_flipping_candidates(densities, flipping_threshold)
            flipping_candidate_rate.append(len(flipping_candidates) / len(densities))
            logging.info(f'\nNumber of fixations being flipping candidates: {len(flipping_candidates)}/{len(densities)}\n')
    return flipping_candidate_rate


def fcr_of_vis_type(vis_type, dataset_dir, flipping_threshold):
    rates = []
    for vis_densities in glob(os.path.join(dataset_dir, 'densitiesByVis', vis_type, '*')):
        #Flipping candidate ratios of all recordings associated to vis
        fc_rate = fcr_of_vis(vis_densities, flipping_threshold)
        if len(fc_rate) > 0:
            avg_fc_rate = np.mean(fc_rate)
            rates.append(avg_fc_rate)
            logging.info(f'\nAverage flipping candidate ratio: {avg_fc_rate:.5f}\n')
        else:
            logging.info('\nNo flipping candidates!\n')
    return rates


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')
    sns.set()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    logging.basicConfig(filename='flipping_rate_analysis.log', filemode='w', level=logging.INFO)
    type2rate = {vt: [] for vt in vis_types}

    for fc_threshold in np.linspace(0, 1, 100):
        for vis_type in args['vis_types']:
            type_rates = fcr_of_vis_type(vis_type, args['dataset_dir'], fc_threshold)
            type2rate[vis_type].append(np.mean(type_rates))

    for vis_type, rates in type2rate.items():
        plt.plot(np.linspace(0, 1, 100), rates, label=vis_type)

    plt.xlabel('Flipping candidate threshold')
    plt.ylabel('Flipping candidate rate')
    plt.title('Flipping candidate rate per vis type')
    plt.legend()
    plt.grid(axis='both')
    plt.show()
