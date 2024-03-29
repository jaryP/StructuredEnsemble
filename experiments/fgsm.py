import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def perturbe_image(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def perturbed_predictions(method, dataloder, epsilon, device, normalize=False):
    if epsilon > 0:
        images = []
        targets = []
        method.train()

        for data, target in dataloder:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            p, _ = method.predict_proba(data, target, True)

            loss = torch.nn.functional.cross_entropy(p, target)

            method.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = perturbe_image(data, epsilon, data_grad)

            images.extend(perturbed_data)
            targets.extend(target)

        images = torch.stack(images, 0)
        targets = torch.stack(targets, 0)

        dataset = TensorDataset(images, targets)
        dataset = DataLoader(dataset, batch_size=dataloder.batch_size)
    else:
        dataset = dataloder

    probs = []
    predictions = []
    true = []
    hs = []

    method.eval()

    for x, y in dataset:
        true.extend(y.tolist())
        p, _ = method.predict_proba(x, y, True)

        plog = (p + 1e-12).log()
        h = plog * p
        if normalize:
            h = h / np.log(h.shape[-1])
        h = -torch.sum(h, -1)

        hs.extend(h.tolist())

        probs.extend(p.tolist())
        pred = torch.argmax(p, -1)
        predictions.extend(pred.tolist())

    predictions = np.asarray(predictions)
    true = np.asarray(true)
    probs = np.asarray(probs)
    hs = np.asarray(hs)

    return true, predictions, probs, hs
