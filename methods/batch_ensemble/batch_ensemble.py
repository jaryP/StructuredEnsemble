import os
import warnings
from copy import deepcopy

import torch

from eval import eval_method, eval_model
from methods.batch_ensemble.layers import layer_to_masked
from methods.base import EnsembleMethod
from utils import train_model


class BatchEnsemble(EnsembleMethod):
    # TODO: implementare inizializzazione come nel paper :
    #                       However, we find it sufficient for BatchEnsemble to have desireddiversity by initializing
    #                       the fast weight (siandriin Eqn. 1) to be random sign vectors.
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']

        self.model = deepcopy(model)
        layer_to_masked(self.model, ensemble=self.ensemble)

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        optim = optimizer([param for name, param in self.model.named_parameters() if param.requires_grad])
        train_scheduler = scheduler(optim)

        assert train_dataset.batch_size >= self.ensemble
        if train_dataset.batch_size % self.ensemble != 0:
            warnings.warn("Batch_size % ensemble != 0 ({} % {} = {}). Custom padding will be used."
                          .format(train_dataset.batch_size, self.ensemble,
                                  train_dataset.batch_size % self.ensemble))

        best_model, scores, best_model_scores, losses = train_model(model=self.model, optimizer=optim,
                                                                    epochs=epochs, train_loader=train_dataset,
                                                                    scheduler=train_scheduler,
                                                                    early_stopping=early_stopping,
                                                                    test_loader=test_dataset, eval_loader=eval_dataset,
                                                                    device=self.device)

        self.model.load_state_dict(best_model)

        return [scores]

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        bs = x.shape[0]
        x = torch.cat([x for _ in range(self.ensemble)], dim=0)
        outputs = self.model(x)
        outputs = torch.split(outputs, bs)
        outputs = torch.stack(outputs, 1)
        if reduce:
            outputs = torch.mean(outputs, 1)
        return outputs

    def load(self, path):
        self.model = torch.load(os.path.join(path, 'model.pt'),
                                map_location=self.device)
        # self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.model, os.path.join(path, 'model.pt'))
