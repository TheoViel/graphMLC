# From https://github.com/HCPLab-SYSU/SSGRL/

import numpy as np


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc12_mAP(preds_voc, num=20, return_aps=False):
    np.savetxt("../output/preds.txt", preds_voc)
    with open("../output/preds.txt", "r") as f:
        lines = f.readlines()

    seg = np.array([x.strip().split(" ") for x in lines]).astype(float)
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)
    threshold = 1 / (num_target + 1e-6)

    predict_result = seg[:, 0:num] > threshold

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    per_class_recall = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = sorted_label[i] > 0
            fp[i] = sorted_label[i] <= 0
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    mAP = np.mean(aps)
    if return_aps:
        return aps
    else:
        return mAP

