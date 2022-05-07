import numpy as np
import json

MAX_WIDTH = [-1, 1066.666, 1169, 1069.25, 1600, 1066.666, 1066.666, 1023.28, 1066.666, 1142.67, 1035.22]
MAX_HEIGHT = [-1, 800, 800, 800, 774.98, 800, 800, 800, 800, 800, 800]

def compute_scale_factor(im, groupID):
    w, h = im.size
    # print(w, h, groupID)
    if MAX_HEIGHT[groupID] / h < MAX_WIDTH[groupID] / w:
        scale_factor = MAX_HEIGHT[groupID] / h
    else:
        scale_factor = MAX_WIDTH[groupID] / w
    return scale_factor

def parse_densities(file):
    return list(json.load(file).values())

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


#Sequence Score
def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0

def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
