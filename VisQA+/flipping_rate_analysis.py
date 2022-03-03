"""
Step 6: Calculates flipping candidate rate (fcr) of each visualization type
Requires precomputed densities for each visualization (Step 4-5), which can be obtained from 'kde_densities.py'
"""

import numpy as np
import os.path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from tqdm import tqdm


def dist_hellinger(densities, N):
    """
    Hellinger distance between uniform distribution of fixed N and the given densities.
    See: https://en.wikipedia.org/wiki/Hellinger_distance#Properties
    """
    uniform = 1 / N
    normalizing = np.sqrt(1 - np.sqrt(.5))

    bhatt_coeff = sum([np.sqrt(d * uniform) for _, d in densities])
    bhatt_coeff = np.clip(bhatt_coeff, 0, 1)

    dist = np.sqrt(1 - bhatt_coeff)
    return dist / normalizing


def dist_total_variation(densities, N):
    uniform = 1 / N
    return sum([abs(d - uniform) for _, d in densities])


def parse_densities(file):
    return json.load(file).values()


def flipping_candidate_score_of_rank(densities, r, dist_fn=dist_hellinger):
    """
    Flipping candidates score has the following interpretation:
    ~> 0: The density distribution is peaked, i.e. the fixation mostly covers just a single AOI.
    ~> 1: The density distribution is close to uniform, i.e. the fixation covers at least two AOI to a very similar extent.

    NOTE: Is there off-the-shelf solution for this? There might be better / more elegant way to compute this.
    """
    N = len(densities)
    copy = list(densities)

    # Add dummy zeros when rank is larger than number density entries.
    if r > N:
        copy.extend([('0', 0)] * (r - N))

    copy.sort(reverse=True, key=lambda x: x[1])
    dist = dist_fn(copy[:r], r)
    return 1. - dist


def find_flipping_candidates(fixation_densities, threshold, target_ranks=(2, 3, 4)):
    """
    Perform KDE analysis steps to find flipping candidates.
    Output can be verified with show_density_overlay=True
    """
    flipping_candidates = []
    for densities in fixation_densities:
        # Step 6: check for which segments the distribution overlays at least two AOIs to a very similar extent (the flipping candidates)
        rank_scores = {2: flipping_candidate_score_of_rank(densities, r=2),
                       3: flipping_candidate_score_of_rank(densities, r=3),
                       4: flipping_candidate_score_of_rank(densities, r=4)}
        rank_of_max = max(rank_scores, key=rank_scores.get)
        if rank_scores[rank_of_max] > threshold and rank_of_max in target_ranks:
            flipping_candidates.append((densities))
    return flipping_candidates


def fcr_of_vis(densities_dir, flipping_threshold, target_ranks):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    flipping_candidate_rate = []
    for path in glob(os.path.join(densities_dir, '*.json')):
        with open(path, 'r') as f:
            densities = parse_densities(f)
            flipping_candidates = find_flipping_candidates(densities, flipping_threshold, target_ranks)
            flipping_candidate_rate.append(len(flipping_candidates) / len(densities))
            #print(f'\nNumber of fixations being flipping candidates: {len(flipping_candidates)}/{len(densities)}\n')
    return flipping_candidate_rate


def fcr_of_vis_type(vis_type, dataset_dir, flipping_threshold, target_ranks):
    rates = []
    for vis_densities in glob(os.path.join(dataset_dir, 'densitiesByVis', vis_type, '*')):
        #Flipping candidate ratios of all recordings associated to vis
        fc_rate = fcr_of_vis(vis_densities, flipping_threshold, target_ranks)
        if len(fc_rate) > 0:
            avg_fc_rate = np.mean(fc_rate)
            rates.append(avg_fc_rate)
            #print(f'\nAverage flipping candidate ratio: {avg_fc_rate:.5f}\n')
    return rates


def threshold_analysis(args, steps):
    thresholds = np.linspace(0, 1, steps)
    threshold_rates = []

    for fc_threshold in tqdm(thresholds):
        rates = []
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', '*', '*')):
            rate = fcr_of_vis(vis_densities, fc_threshold, target_ranks=(2, ))
            rates.extend(rate)
        threshold_rates.append(np.mean(rates))

    plt.plot(thresholds, threshold_rates, label='rank 2')
    plt.xlabel('Flipping candidate threshold')
    plt.ylabel('Flipping candidate rate')
    plt.show()


def type_analysis(args, vis_types, fc_threshold):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    type2rate = {vt: [] for vt in vis_types}

    for n, vis_type in enumerate(args['vis_types']):
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            rate = fcr_of_vis(vis_densities, fc_threshold, target_ranks=(2,))
            type2rate[vis_type].extend(rate)
    sns.boxplot(data=list(type2rate.values()), showfliers=False)
    #ax.violinplot(type2rate.values(), showmeans=True, showmedians=False, showextrema=False)

    ax.set_xticklabels(type2rate.keys())
    ax.set_xticks(np.arange(len(type2rate)))
    ax.set_ylabel('Flipping candidate rate')
    ax.set_xticklabels(type2rate.keys())
    plt.show()


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')
    sns.set()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    threshold_analysis(args, 10)
    #type_analysis(args, vis_types, 0.2)
