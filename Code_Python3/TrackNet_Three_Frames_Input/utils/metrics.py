import numpy as np
def classify_metrics(pred, gt):
    seqlen = gt.shape[0]
    TP = 0
    ALL_HAS = 0
    FP = 0
    diff = 0
    for j in range(seqlen):
        if gt[j] == 1:
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            ALL_HAS += 1
            if 1 in pred[start:end]:
                TP += 1
                ind = start+np.where(pred[start:end]==1)[0][0]
                diff += abs(ind-j)
        if pred[j] == 1:
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            if 1 not in gt[start:end]:
                FP += 1
    return TP, ALL_HAS, FP, diff