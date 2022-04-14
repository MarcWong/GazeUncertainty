"""
Step 7: Altering flipping candidate and calculate the Sequence Score for each visualization type
Requires precomputed densities for each visualization (Step 4-5), which can be obtained from 'kde_densities.py'
"""

import numpy as np
import pandas as pd
import os.path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from tqdm import tqdm
from util import nw_matching, parse_densities, find_flipping_candidates
from scipy.stats import ttest_ind

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
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    type2rate = {vt: [] for vt in vis_types}
    type2ratebad = {vt: [] for vt in vis_types}

    for n, vis_type in enumerate(args['vis_types']):
        for vis_densities in glob(os.path.join(args['dataset_dir'], 'densitiesByVis', vis_type, '*')):
            rate, rate_bad = SS_of_vis_groups(vis_densities, fc_threshold, target_ranks=(2,))
            type2rate[vis_type].extend(rate)
            type2ratebad[vis_type].extend(rate_bad)

    for axis in ax:
        axis.set(ylim=(0.4,1.02))
        axis.axhline(y=0.5, color='#BBBBBB', linestyle='--')
        axis.axhline(y=0.6, color='#BBBBBB', linestyle='--')
        axis.axhline(y=0.7, color='#BBBBBB', linestyle='--')
        axis.axhline(y=0.8, color='#BBBBBB', linestyle='--')
        axis.axhline(y=0.9, color='#BBBBBB', linestyle='--')
        axis.axhline(y=1.0, color='#BBBBBB', linestyle='--')


    sns.boxplot(data=list(type2rate.values()), showfliers=False, ax=ax[0])
    sns.swarmplot(data=list(type2rate.values()), color=".25", ax=ax[0])
    sns.boxplot(data=list(type2ratebad.values()), showfliers=False, ax=ax[1])
    sns.swarmplot(data=list(type2ratebad.values()), color=".25", ax=ax[1])

    print(len(type2rate['bar']), len(type2ratebad['bar']))
    print(len(type2rate['line']), len(type2ratebad['line']))
    print(len(type2rate['scatter']), len(type2ratebad['scatter']))
    print(len(type2rate['pie']), len(type2ratebad['pie']))
    out_bar = ttest_ind(type2rate['bar'], type2ratebad['bar'], equal_var=False)
    out_line = ttest_ind(type2rate['line'], type2ratebad['line'], equal_var=False)
    out_scatter = ttest_ind(type2rate['scatter'], type2ratebad['scatter'], equal_var=False)
    out_pie = ttest_ind(type2rate['pie'], type2ratebad['pie'], equal_var=False)

    print(f'bar: {out_bar}')
    print(f'line: {out_line}')
    print(f'scatter: {out_scatter}')
    print(f'pie: {out_pie}')

    ax[0].set_xticklabels(type2rate.keys())
    ax[0].set_xticks(np.arange(len(type2rate)))
    ax[0].set_xlabel('Low calibration error')
    ax[1].set_xticklabels(type2ratebad.keys())
    ax[1].set_xticks(np.arange(len(type2ratebad)))
    ax[1].set_xlabel('High calibration error')
    ax[0].set_ylabel('Sequence Score')

    plt.show()

if __name__ == '__main__':
    VIS_TYPES = ('bar', 'line', 'scatter', 'pie')
    #VIS_TYPES = ('bar', 'line', 'scatter', 'pie', 'table', 'other')
    sns.set_theme(style="white", font_scale=2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--vis_types", choices=VIS_TYPES, nargs='+', default=VIS_TYPES)
    args = vars(parser.parse_args())
    vis_types = set(args['vis_types'])

    type_analysis(args, vis_types, 0.2)
