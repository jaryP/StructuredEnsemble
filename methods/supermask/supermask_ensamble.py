import logging

import torch
from tqdm import tqdm
import numpy as np
from .base import layer_to_masked, EnsembleMaskedWrapper, extract_distribution_subnetwork
from .trainable_masks import MMD
from eval import eval_model

import matplotlib.pyplot as plt
import seaborn as sns


def EnsembleSupermaskBeforeTraining(model, train_dataset, test_dataset, mask_epochs, prune_percentage, ensemble=1,
                                    re_init=True, global_pruning=True, device='cpu', eval_dataset=None, supermask=None,
                                    **kwargs):

    logger = logging.getLogger(__name__ + '.' + EnsembleSupermaskBeforeTraining.__name__)
    logger.info('Applying the weights to the model')
    layer_to_masked(model, masks_params=supermask, ensemble=ensemble)
    logger.info('New model created')
    model.to(device)

    # print(prune_percentage)
    # if prior is not None:
    # divergence_w = prior.get('divergence_w', 1e-4)
    # prior = get_distribution(**prior)

    #     fig = plt.figure('prior')
    #     fig.suptitle('prior', fontsize=20)
    #     d = prior.sample(sample_shape=[1000]).cpu().numpy()
    #     ax = sns.kdeplot(data=d)

    # if binary_mask:

    # for name, module in model.named_modules():
    #     if isinstance(module, EnsembleMaskedWrapper):
    #         distr = module.distributions
    #         # print(len(distr))
    #         for i, distribution in enumerate(distr):
    #             fig = plt.figure(num=name)
    #             ax = fig.add_subplot(2, len(distr), i+1)
    #             fig.suptitle(name, fontsize=20)
    #             d = distribution(size=50, reduce=False).detach().cpu()
    #             d = d.view(d.size(0), -1).numpy()
    #             sns.barplot(data=d, errwidth=0.1, ax=ax)
    #             ax.set_xticks([])
    #         # break

    bar = tqdm(range(mask_epochs), desc='Mask: {}'.format('Mask training'))

    # with torch.no_grad():
    #     for name, module in model.named_modules():
    #         if isinstance(module, MaskedLayerWrapper):
    #             fig = plt.figure('initial_'+name)
    #             fig.suptitle('initial_'+name, fontsize=20)
    #             d = module.distribution(size=50, reduce=False).cpu()
    #             d = d.view(d.size(0), -1).numpy()
    #             ax = sns.barplot(data=d, errwidth=0.1)
    #             ax.set_xticks([])
    #             ax.set_ylim(0, 1)
    #             # break

    masks_parameters = [param for name, param in model.named_parameters() if 'distribution' in name]
    optim = torch.optim.Adam(masks_parameters, lr=0.001)
    # optim = optimizer(masks_parameters)

    # pps = []
    # if binary_mask:
    #     pps = [0] * (mask_epochs // 2) + list(np.linspace(0, prune_percentage, mask_epochs // 2))

    for e in bar:
        losses = []
        model.train()

        for i, (x, y) in enumerate(train_dataset):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

            kl = torch.tensor(0.0, device=device)

            if ensemble > 1:
                for name, module in model.named_modules():
                    if isinstance(module, EnsembleMaskedWrapper):
                        distr = module.distributions

                        # if len(distr) > 1:
                        # mmd_matrix = torch.empty((len(distr), len(distr))).fill_(0)

                        _mmd = torch.tensor(0.0, device=device)
                        for i, d1 in enumerate(distr):
                            for j in range(i+1, len(distr)):
                                # _mmd = KL(d1.posterior, d2.posterior) / x.size(0)
                                _mmd += MMD(d1, distr[j])
                                # mmd_matrix[i][j] = _mmd

                        kl += _mmd
                kl = 1 / kl
                # kl += torch.triu(mmd_matrix).sum() / len(distr)
                kl *= 1e-2

            losses.append((loss.item(), kl.item()))

            loss += kl

            optim.zero_grad()
            loss.backward()
            optim.step()

        bar.set_postfix({'losses': np.mean(losses, 0)})

        if eval_dataset is not None:
            eval_scores = eval_model(model, eval_dataset, device=device, topk=[1, 5])
        else:
            eval_scores = 0

        train_scores = eval_model(model, train_dataset, device=device)
        test_scores = eval_model(model, test_dataset, device=device)

        logger.info('Epoch {}/{} over. Results:'.format(e + 1, mask_epochs))
        logger.info('\tTrain: {}'.format(train_scores))
        logger.info('\tEval: {}'.format(eval_scores))
        logger.info('\tTest: {}'.format(test_scores))

    # for name, module in model.named_modules():
    #     if isinstance(module, EnsembleMaskedWrapper):
    #         distr = module.distributions
    #         # print(len(distr))
    #         for i, distribution in enumerate(distr):
    #             fig = plt.figure(num=name)
    #             ax = fig.add_subplot(2, len(distr), len(distr)+i+1)
    #             fig.suptitle(name, fontsize=20)
    #             d = distribution(size=50, reduce=False).detach().cpu()
    #             d = d.view(d.size(0), -1).numpy()
    #             sns.barplot(data=d, errwidth=0.1, ax=ax, )
    #             ax.set_xticks([])
    #
    # plt.show()

    # model.detach().cpu()
    # s1 = {name: module.weight for name, module in model.named_modules() if hasattr(module, 'weight')}
    # s = sum(w.numel() for _, w in s1.items())
    models = []
    # print('Original size: ', s)
    for i in range(ensemble):
        m = extract_distribution_subnetwork(model, prune_percentage, i,
                                            re_init=re_init, global_pruning=global_pruning)
        # print(i)
        # print(m)
        # s1 = {name: module.weight for name, module in m.named_modules() if hasattr(module, 'weight')}
        # s = sum(w.numel() for _, w in s1.items())
        # print(s)
        # print('#'*100)
        models.append(m)

    return models
