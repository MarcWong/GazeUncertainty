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
import pandas as pd

from glob import glob
from tqdm import tqdm
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
    return list(json.load(file).values())


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


def fcr_of_vis(densities_dir, flipping_threshold, target_ranks):
    """
    Returns rate of flipping candidates of all recordings associated to given visualization.
    """
    flipping_candidate_rate = []
    for path in glob(os.path.join(densities_dir, '*.json')):
        with open(path, 'r') as f:
            densities = parse_densities(f)
            flipping_candidates, _ = find_flipping_candidates(densities, flipping_threshold, target_ranks)
            flipping_candidate_rate.append(len(flipping_candidates) / len(densities))
            #print(f'\nNumber of fixations being flipping candidates: {len(flipping_candidates)}/{len(densities)}\n')
    return flipping_candidate_rate


def fcr_of_vis_type(vis_type, dataset_dir, flipping_threshold, target_ranks):
    rates = []
    for densities_dir in glob(os.path.join(dataset_dir, 'densitiesByVis', vis_type, '*')):
        #Flipping candidate ratios of all recordings associated to vis
        fc_rate = fcr_of_vis(densities_dir, flipping_threshold, target_ranks)
        if len(fc_rate) > 0:
            avg_fc_rate = np.mean(fc_rate)
            rates.append(avg_fc_rate)
            #print(f'\nAverage flipping candidate ratio: {avg_fc_rate:.5f}\n')
    return rates


def plot_fcr_threshold(args, threshold_steps, fc_ranks):
    thresholds = np.linspace(0, 1, threshold_steps)
    all_fcr = []

    for fc_threshold in tqdm(thresholds):
        fcr = []
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', '*', '*')):
            # FCR of each subject
            vis_fcr = fcr_of_vis(vis_densities, fc_threshold, target_ranks=fc_ranks)
            # For each vis, we calculate the average FCR among all subjects.
            avg_fcr = np.mean(vis_fcr)
            fcr.append(avg_fcr)
        # For each type, we calculate the average FCR among all vis
        all_fcr.append(np.mean(fcr))

    plt.plot(thresholds, all_fcr)

    plt.xlabel('Flipping candidate threshold')
    plt.ylabel('Flipping candidate rate')
    plt.show()


def plot_fcr_distribution(args, vis_types, threshold, fc_ranks):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    type2fcr = {vt: [] for vt in vis_types}

    for n, vis_type in enumerate(args['vis_types']):
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            # FCR of each subject
            vis_fcr = fcr_of_vis(vis_densities, threshold, target_ranks=fc_ranks)
            # For each vis, we calculate the average FCR among all subjects.
            avg_fcr = np.mean(vis_fcr)
            type2fcr[vis_type].append(avg_fcr)
    
    out_bar_scatter = ttest_ind(type2fcr['bar'], type2fcr['scatter'], equal_var=False)
    out_bar_line = ttest_ind(type2fcr['bar'], type2fcr['line'], equal_var=False)
    out_line_scatter = ttest_ind(type2fcr['line'], type2fcr['scatter'], equal_var=False)

    print(f'bar, scatter: {out_bar_scatter}')
    print(f'bar, line: {out_bar_line}')
    print(f'line, scatter: {out_line_scatter}')
    sns.boxplot(data=list(type2fcr.values()))
    sns.swarmplot(data=list(type2fcr.values()), color=".25")

    ax.set_xticklabels(type2fcr.keys())
    ax.set_xticks(np.arange(len(type2fcr)))
    ax.set_ylabel('Flipping candidate rate')
    ax.set_xticklabels(type2fcr.keys())
    plt.show()


def aoi_proportion_in_fc(args, vis_type, fc_threshold):
    """
    Calculates the proportion of aoi pairs occuring in flipping candidates of rank 2.
    """
    pair2ratio = {}
    for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
        for path in glob(os.path.join(vis_densities, '*.json')):
            with open(path, 'r') as f:
                pair2count = {}
                densities = parse_densities(f)
                flipping_candidates, _ = find_flipping_candidates(densities, fc_threshold, target_ranks=(2,))
                # Count aoi pair occurrences in flipping candidates
                for c in flipping_candidates:
                    if len(c) < 2:
                        continue
                    # These are the two aoi associated to the greatest two densities
                    (aoi_1, _), (aoi_2, _) = sorted(c, key=lambda x: x[1], reverse=True)[:2]
                    pair_id = aoi_1 + aoi_2
                    pair2count[pair_id] = pair2count.get(pair_id, 0) + 1

                # Normalize the candidate counts to get ratio
                for aoi_pair, cnt in pair2count.items():
                    ratio = cnt / len(flipping_candidates)
                    pair2ratio[aoi_pair] = pair2ratio.get(aoi_pair, 0) + ratio

    # The order of labels in an aoi pair is irrelevant, e.g. TX = XT
    for pair_id in sorted(pair2ratio.keys()):
        reversed = pair_id[::-1]
        if pair_id != reversed and reversed in pair2ratio:
            pair2ratio[pair_id] += pair2ratio[reversed]
            pair2ratio[reversed] = 0

    # Average ratios
    total_count = sum(pair2ratio.values())
    return {k: v / total_count for k, v in pair2ratio.items() if v > 0}


def plot_aoi_proportion_in_fc(args, vis_types, threshold):
    fig, ax = plt.subplots(nrows=1, ncols=len(vis_types))
    df_merged = pd.DataFrame()

    for vis_type in vis_types:
        aoi_pair_cnt = aoi_proportion_in_fc(args, vis_type, threshold)
        df = pd.DataFrame(aoi_pair_cnt.items(), columns=['aoi_pair', 'freq'])
        df['vis_type'] = vis_type
        df_merged = df_merged.append(df)
    aoi_pairs = df_merged['aoi_pair'].unique()
    pal = dict(zip(aoi_pairs, sns.color_palette("colorblind", n_colors=len(aoi_pairs))))

    for n, vis_type in enumerate(vis_types):
        df = df_merged[df_merged['vis_type'] == vis_type].sort_values('freq')
        axs = sns.barplot(x='aoi_pair', y='freq', data=df, ax=ax[n], palette=pal)
        axs.set_xticklabels(axs.get_xticklabels(), rotation=0)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.set_title(vis_type)
        axs.set_ylim(0, 1)

    plt.show()


def merge_prob_by_aoi(flipping_candidates):
    merged_prob = {}
    for fpc in flipping_candidates:
        merged_prob[fpc[0]] = merged_prob.get(fpc[0], 0) + fpc[1]
    return sorted(merged_prob.items(), key=lambda kv:(kv[1], kv[0]))


def fc_proportion_on_first(args, fc_threshold, fc_ranks):
    """
    Calculates the propertion of first fixations being flipping candidates
    """
    scanpath = []
    type2ratio = {vt: [] for vt in args['vis_types']}
    for vis_type in args['vis_types']:
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            cnt_subjects = cnt_first = 0
            for path in glob(os.path.join(vis_densities, '*.json')):
                cnt_subjects += 1
                with open(path, 'r') as f:
                    fixation_densities = parse_densities(f)
                    flipping_candidates, idxs = find_flipping_candidates(fixation_densities, fc_threshold, fc_ranks)
                    merged = merge_prob_by_aoi(flipping_candidates)

                    if len(merged) < 2:
                        aoi_1, aoi_2 = merged[-1][0], merged[-1][0]
                    else:
                        aoi_1, aoi_2 = merged[-1][0], merged[-2][0]

                    scanpath.ap

            # Propertion of first fixations being flipping candidates on vis
            ratio_first = cnt_first / cnt_subjects
            type2ratio[vis_type].append(ratio_first)
    # Averaged over each vis
    return {vt: np.mean(type2ratio[vt]) for vt in args['vis_types']}


def plot_fc_proportion_on_first(args, threshold_steps, fc_ranks):
    thresholds = np.linspace(0, 1, threshold_steps)
    all_type2ratio = {vt: [] for vt in args['vis_types']}

    for fc_threshold in tqdm(thresholds):
        type2ratio = fc_proportion_on_first(args, fc_threshold, fc_ranks)
        all_type2ratio = {vt: all_type2ratio[vt] + [type2ratio[vt]] for vt in args['vis_types']}
    
    for vis_type in args['vis_types']:
        plt.plot(thresholds, all_type2ratio[vis_type], label=vis_type)
    plt.xlabel('Flipping candidate threshold')
    plt.ylabel('Proportion of first fixations being flipping candidates')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')
    sns.set_theme(style="white", font_scale=2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    #plot_fcr_threshold(args, threshold_steps=100, fc_ranks=(2, 3, 4))

    # TODO Find "reasonable" threshold
    plot_fcr_distribution(args, vis_types, threshold=0.5, fc_ranks=(2, 3, 4))

    #plot_fc_proportion_on_first(args, threshold_steps=100, fc_ranks=(2, 3, 4))

    # Currently we analysis on FC of rank 2, i.e. analysing aoi pairs occuring in densities.
    plot_aoi_proportion_in_fc(args, vis_types, threshold=0.5)
