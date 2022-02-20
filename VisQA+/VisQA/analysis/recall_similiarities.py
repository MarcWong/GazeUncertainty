import numpy as np
import argparse
import math

from PIL import Image
from glob import glob
from natsort import natsorted


def auc_saliconeval(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    Sth = np.asarray([salMap[y-1][x-1] for x, y in gtsAnn])

    Nfixations = len(gtsAnn)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    np.concatenate((Sth, randfix), axis=0)
    allthreshes = np.arange(
        0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) /
                Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(randfix >= thresh)) /
                Nrand for thresh in allthreshes]
    auc = np.trapz(tp, fp)
    return auc


def cc_npy(gt, predicted):
    M1 = np.divide(predicted - np.mean(predicted), np.std(predicted))
    M2 = np.divide(gt - np.mean(gt), np.std(gt))
    ret = np.corrcoef(M1.reshape(-1), M2.reshape(-1))[0][1]
    return ret


def calc_similities_between_heat_maps(plot_overall_dir: str):
    correct = natsorted(glob(plot_overall_dir + '/*_correct_sum_hm.png'))
    incorrect = natsorted(glob(plot_overall_dir + '/*_incorrect_sum_hm.png'))

    ccs = []
    aucs = []
    for c, i in zip(correct, incorrect):
        c, i = np.asarray(Image.open(c)), np.asarray(Image.open(i))
        ccs.append(cc_npy(c, i))
        # aucs.append(auc_saliconeval(c, i))

    ccs = [value for value in ccs if not math.isnan(value)]
    cc = np.mean(ccs)
    auc = np.mean(aucs)
    print("CC ")
    print(cc)
    print("AUC ")
    print(auc)
    return cc, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_overall", type=str, default=None)
    args = vars(parser.parse_args())
    calc_similities_between_heat_maps(args['plot_overall'])
