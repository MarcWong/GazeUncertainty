"""
Step 6: Calculates flipping candidate rate (fcr) of each visualization type
Requires precomputed densities for each visualization (Step 4-5), which can be obtained from 'kde_densities.py'
"""

import numpy as np
import pandas as pd
import os.path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from tqdm import tqdm
from util import nw_matching
from scipy.stats import ttest_ind


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
    #return sum([abs(d - uniform) for _, d in densities])
    t1 = sum([abs(d - uniform) for _, d in densities])
    t2 = max(1 - sum([d for _, d in densities]), 0)

    return (t1 + t2)

def parse_densities(file):
    return json.load(file).values()

def parse_scanpath(densities):
    label = ''
    for pos in densities:
        highest_prob = 0
        tmplabel = ''
        for candidate in pos:
            #candidate[0] is AOI label, and candidate[1] is prob
            if candidate[1] > highest_prob:
                tmplabel = candidate[0]
                highest_prob = candidate[1] 
        label += tmplabel
    return label

def flipping_candidate_score_of_rank(densities, r, dist_fn=dist_total_variation):
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
    idxs = []
    for idx, densities in enumerate(fixation_densities):
        # Step 6: check for which segments the distribution overlays at least two AOIs to a very similar extent (the flipping candidates)
        rank_scores = {2: flipping_candidate_score_of_rank(densities, r=2),
                       3: flipping_candidate_score_of_rank(densities, r=3),
                       4: flipping_candidate_score_of_rank(densities, r=4)}
        rank_of_max = max(rank_scores, key=rank_scores.get)
        if rank_scores[rank_of_max] > threshold and rank_of_max in target_ranks:
            flipping_candidates.append((densities))
            idxs.append((idx))
    return flipping_candidates, idxs

def alter_candidate(flipping_candidates):
    merged_prob = {}
    for fpc in flipping_candidates:
        merged_prob[fpc[0]] = merged_prob.get(fpc[0], 0) + fpc[1]

    merged_prob = sorted(merged_prob.items(), key = lambda kv:(kv[1], kv[0]))
    if(len(merged_prob) < 2):
        return merged_prob[-1][0], merged_prob[-1][0]
    # new AOI, old AOI
    return merged_prob[-2][0], merged_prob[-1][0]

def alter_scanpath(scanpath_bs, idxs, flipping_candidates):
    scanpath_new = scanpath_bs
    for i in range(len(flipping_candidates)):
        NewC, OldC = alter_candidate(flipping_candidates[i])
        scanpath_new = scanpath_new[:idxs[i]] + NewC + scanpath_new[idxs[i]+1:]
    return scanpath_new

def SS_of_vis_groups(densities_dir, flipping_threshold, target_ranks):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    SS_good = []
    SS_bad = []

    df_img_excluded = pd.read_csv(
        'dataset/excluded.csv')

    for fix_path in glob(os.path.join(densities_dir, '*.json')):
        with open(fix_path, 'r') as f:
            densities = parse_densities(f)
            scanpath_bs = parse_scanpath(densities)
            #print('scanpath length:',len(densities))
            flipping_candidates, idxs = find_flipping_candidates(densities, flipping_threshold, target_ranks)
            scanpath_new = alter_scanpath(scanpath_bs, idxs, flipping_candidates)
            subject_id = fix_path.split('/')[-1].strip('.json')
            if subject_id in df_img_excluded['subject_id'].values:
                SS_bad.append(nw_matching(scanpath_bs, scanpath_new))  
            else:
                SS_good.append(nw_matching(scanpath_bs, scanpath_new))
    return SS_good, SS_bad


def SS_of_vis(densities_dir, flipping_threshold, target_ranks):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    SS = []
    for path in glob(os.path.join(densities_dir, '*.json')):
        with open(path, 'r') as f:
            densities = parse_densities(f)
            scanpath_bs = parse_scanpath(densities)
            #print('scanpath length:',len(densities))
            flipping_candidates, idxs = find_flipping_candidates(densities, flipping_threshold, target_ranks)
            scanpath_new = alter_scanpath(scanpath_bs, idxs, flipping_candidates)
            SS.append(nw_matching(scanpath_bs, scanpath_new))
    return SS


def type_analysis(args, vis_types, fc_threshold):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    type2rate = {vt: [] for vt in vis_types}
    type2ratebad = {vt: [] for vt in vis_types}

    for n, vis_type in enumerate(args['vis_types']):
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            rate, rate_bad = SS_of_vis_groups(vis_densities, fc_threshold, target_ranks=(2,))
            type2rate[vis_type].extend(rate)
            type2ratebad[vis_type].extend(rate_bad)

    sns.boxplot(data=list(type2rate.values()), showfliers=False, ax=ax1)
    sns.swarmplot(data=list(type2rate.values()), color=".25", ax=ax1)
    sns.boxplot(data=list(type2ratebad.values()), showfliers=False, ax=ax2)
    sns.swarmplot(data=list(type2ratebad.values()), color=".25", ax=ax2)

    print(len(type2rate['bar']), len(type2ratebad['bar']))
    print(len(type2rate['line']), len(type2ratebad['line']))
    print(len(type2rate['scatter']), len(type2ratebad['scatter']))
    out_bar = ttest_ind(type2rate['bar'], type2ratebad['bar'], equal_var=False)
    out_line = ttest_ind(type2rate['line'], type2ratebad['line'], equal_var=False)
    out_scatter = ttest_ind(type2rate['scatter'], type2ratebad['scatter'], equal_var=False)

    print(f'bar: {out_bar}')
    print(f'line: {out_line}')
    print(f'scatter: {out_scatter}')

    ax1.set_xticklabels(type2rate.keys())
    ax1.set_xticks(np.arange(len(type2rate)))
    ax1.set_xlabel('Low calibration error')
    ax1.set_ylabel('Sequence Score')
    ax1.set(ylim=(0.4,1))

    ax2.set_xticklabels(type2ratebad.keys())
    ax2.set_xticks(np.arange(len(type2ratebad)))
    ax2.set_xlabel('High calibration error')
    ax2.set(ylim=(0.4,1))
    plt.show()

if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter')
    #VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')
    sns.set_theme(style="white", font_scale=2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    type_analysis(args, vis_types, 0.2)
