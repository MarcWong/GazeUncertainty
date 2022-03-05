import numpy as np

MAX_WIDTH = [-1, 1066.666, 1169, 1069.25, 1600, 1066.666, 1066.666, 1023.28, 1066.666, 1142.67, 1035.22]
MAX_HEIGHT = [-1, 800, 800, 800, 774.98, 800, 800, 800, 800, 800, 800]

def compute_scale_factor(im, groupID):
    w, h = im.size
    print(w, h, groupID)
    if MAX_HEIGHT[groupID] / h < MAX_WIDTH[groupID] / w:
        scale_factor = MAX_HEIGHT[groupID] / h
    else:
        scale_factor = MAX_WIDTH[groupID] / w
    return scale_factor


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
