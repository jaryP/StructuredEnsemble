from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def eval_model(model, dataset, topk=None, device='cpu'):
    model.eval()

    predictions = []
    true = []
    losses = []

    for x, y in dataset:
        true.extend(y.tolist())
        x, y = x.to(device), y.to(device)
        outputs = model(x)

        loss = torch.nn.functional.cross_entropy(outputs, y, reduction='none')
        losses.extend(loss.tolist())

        top_classes = torch.topk(outputs, outputs.size(-1))[1]
        predictions.extend(top_classes.tolist())

    predictions = np.asarray(predictions)
    true = np.asarray(true)

    accuracies = accuracy_score(true, predictions, topk=topk)

    # cm = confusion_matrix(true, predictions)
    # TODO: aggiungere calcolo f1 score

    return accuracies, np.mean(losses)


@torch.no_grad()
def eval_method(method, dataset, topk=None):
    predictions = []
    true = []
    losses = []

    method.eval()

    for x, y in dataset:
        true.extend(y.tolist())
        preds, logits = method.predict_proba(x, y, reduce=True)

        loss = torch.nn.functional.cross_entropy(logits.cpu(), y, reduction='none')
        losses.extend(loss.tolist())

        # preds = method.predict_proba(x, y)
        top_classes = torch.topk(preds, preds.shape[-1])[1]
        predictions.extend(top_classes.tolist())

    predictions = np.asarray(predictions)
    true = np.asarray(true)

    accuracies = accuracy_score(true, predictions, topk=topk)

    # cm = confusion_matrix(true, predictions)
    # TODO: aggiungere calcolo f1 score

    return accuracies, np.mean(losses)


@torch.no_grad()
def get_predictions(method, dataset):

    probs = []
    predictions = []
    true = []
    method.eval()

    for x, y in dataset:
        true.extend(y.tolist())
        p, _ = method.predict_proba(x, y, True)
        probs.extend(p.tolist())
        pred = torch.argmax(p, -1)
        predictions.extend(pred.tolist())

    predictions = np.asarray(predictions)
    true = np.asarray(true)
    probs = np.asarray(probs)

    return true, predictions, probs


def get_logits(method, dataset):
    probs = []
    true = []
    predictions = []

    method.eval()

    for x, y in dataset:
        true.extend(y.tolist())
        p = method.predict_logits(x, y, False)
        probs.extend(p.tolist())

        p = torch.mean(p, 1)
        pred = torch.argmax(p, -1)
        predictions.extend(pred.tolist())

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    probs = np.stack(probs, 0)

    return probs, true, predictions


# def eval_models(model, dataset, topk=None, device='cpu'):
#     if not isinstance(model, list):
#         model = [model]
#
#     for m in model:
#         m.to(device)
#         m.eval()
#
#     predictions = []
#     true = []
#
#     for x, y in dataset:
#         true.extend(y.tolist())
#         x, y = x.to(device), y.to(device)
#         if len(model) > 1:
#             outputs = torch.stack([m(x) for m in model])
#             outputs = torch.mean(outputs, 0)
#         else:
#             outputs = model[0](x)
#
#         top_classes = torch.topk(outputs, outputs.shape[-1])[1]
#         predictions.extend(top_classes.tolist())
#
#     predictions = np.asarray(predictions)
#     true = np.asarray(true)
#
#     accuracies = accuracy_score(true, predictions, topk=topk)
#
#     # cm = confusion_matrix(true, predictions)
#     # TODO: aggiungere calcolo f1 score
#
#     return accuracies
#
#
# @torch.no_grad()
# def eval_model(model, dataset, topk=None, device='cpu'):
#     model.eval()
#     predictions = []
#     true = []
#
#     for x, y in dataset:
#         true.extend(y.tolist())
#         x, y = x.to(device), y.to(device)
#         outputs = model(x)
#         top_classes = torch.topk(outputs, outputs.size(-1))[1]
#         predictions.extend(top_classes.tolist())
#
#     predictions = np.asarray(predictions)
#     true = np.asarray(true)
#
#     accuracies = accuracy_score(true, predictions, topk=topk)
#
#     # cm = confusion_matrix(true, predictions)
#     # TODO: aggiungere calcolo f1 score
#
#     return accuracies


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    assert len(expected) == len(predicted)
    assert predicted.shape[1] >= max(topk)

    res = defaultdict(int)
    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


def confusion_matrix(true: np.asarray, pred: np.asarray, ):
    num_classes = np.max(true) + 1
    pred = pred[:, 0]
    m = np.zeros((num_classes, num_classes), dtype=int)
    for pred, exp in zip(pred, true):
        m[pred][exp] += 1
    return m