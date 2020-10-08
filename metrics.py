# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import numpy as np
import torch
import sklearn.metrics as sk

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_ood_measures(confidences, targets, recall_level=0.95):
    num_inlier = targets.size(0)
    confidences = confidences.data.cpu().numpy()
#     print(confidences)
    pos = np.array(confidences[:num_inlier]).reshape((-1, 1))
    neg = np.array(confidences[num_inlier:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    
    labels = np.ones(len(examples), dtype=np.int32)
    labels[len(pos):] -= 1
#     print(examples, labels)
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_ood_measures_2(known, novel, recall_level=0.95):
#     print(confidences)
    pos = np.array(known).reshape((-1, 1))
    neg = np.array(novel).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    
    labels = np.ones(len(examples), dtype=np.int32)
    labels[len(pos):] -= 1
#     print(examples, labels)
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def tpr95(ind_scores, ood_scores):
    #calculate the falsepositive error when tpr is 95%
    T = 1000
    start=0.01
    end=1
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = ood_scores
    X1 = ind_scores
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fpr = fpr/total
            
def auroc(ind_scores, ood_scores):
    #calculate the AUROC

    # calculate our algorithm
    T = 1000
    start = 0.1
    end = 0.12 
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = ood_scores
    X1 = ind_scores
    aurocNew = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocNew += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocNew += fpr * tpr
    return aurocNew

def auprIn(ind_scores, ood_scores):
    #calculate the AUPR

    # calculate our algorithm
    T = 1000
    start = 0.1
    end = 0.12 
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = ood_scores
    X1 = ind_scores
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        #precisionVec.append(precision)
        #recallVec.append(recall)
        auprNew += (recallTemp-recall)*precision
        recallTemp = recall
    auprNew += recall * precision
    return auprNew

def auprOut(ind_scores, ood_scores):
    #calculate the AUPR
    T = 1000
    start = 0.1
    end = 1 
    gap = (end- start)/100000
    Y1 = ood_scores
    X1 = ind_scores
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    return auprBase

def detection(ind_scores, ood_scores):
    #calculate the minimum detection error

    # calculate our algorithm
    T = 1000
    start = 0.1
    end = 0.12 
#     if name == "CIFAR-100": 
# 	start = 0.01
# 	end = 0.0104    
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = ood_scores
    X1 = ind_scores
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr+error2)/2.0)
            
    return errorNew



def classify_acc_w_ood(logits, targets, confidences, step=1000):
    threshold_min = torch.min(confidences)
    threshold_max = torch.max(confidences)
    threshold_diff = threshold_max - threshold_min
    total = logits.size(0) 

    class_correct = (torch.argmax(logits[:len(targets)], dim=1) == targets).float()
    
    max_threshold = threshold_min
    max_acc = -1.
    for i in range(step + 1):
        threshold = threshold_min + threshold_diff * (i / step)
        inliers = (confidences >= threshold).float()
        outliers = (confidences < threshold).float()
        inlier_correct = (torch.squeeze(inliers[:len(targets)], dim=1) * class_correct[:]).sum()
        outlier_correct = outliers[len(targets):].sum()
        acc = (inlier_correct + outlier_correct) / total
        if max_acc < acc:
            max_acc = acc
            max_threshold = threshold
    
    return max_acc


# Add new metrics here!!!
def show_wrong_samples_targets(logits, targets, log):
    predicts = logits.max(dim=1).indices
    wrong_targets = ((logits.max(dim=1).indices) != targets)
    for idx, i in enumerate(wrong_targets):
        if i:
            log.write("classifier predict [{}] / Ground truth [{}]\n".format(predicts[idx], targets[idx]))