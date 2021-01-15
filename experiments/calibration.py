import numpy as np

from eval import get_predictions


def ece_score(method, dataset, bins=30):

    true, predictions, probs = get_predictions(method, dataset)
    probs = np.max(probs, -1)

    prob_pred = np.zeros((0,))
    prob_true = np.zeros((0,))
    ece = 0

    mce = []

    for b in range(1, int(bins) + 1):
        i = np.logical_and(probs <= b / bins, probs > (b - 1) / bins)  # indexes for p in the current bin

        s = np.sum(i)

        if s == 0:
            prob_pred = np.hstack((prob_pred, 0))
            prob_true = np.hstack((prob_true, 0))
            continue

        m = 1 / s
        acc = m * np.sum(predictions[i] == true[i])
        conf = np.mean(probs[i])

        prob_pred = np.hstack((prob_pred, conf))
        prob_true = np.hstack((prob_true, acc))
        diff = np.abs(acc - conf)

        mce.append(diff)

        ece += (s / len(true)) * diff

    return ece, prob_pred, prob_true, mce
