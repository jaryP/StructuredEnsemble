import os
import pickle
# from copy import deepcopy
from copy import deepcopy

import torch

from methods.base import EnsembleMethod
from methods.snapshot.utils import SnapshotAnnealing
from utils import train_model


class Snapshot(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.method_parameters = method_parameters

        self.device = device

        self.resets = method_parameters.get('n_ensemble', None)
        self.epochs_per_cycle = method_parameters.get('epochs_per_cycle', None)

        self.models = []
        for i in range(self.resets):
            model = deepcopy(model)
            for name, module in model.named_modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            self.models.append(model)

        assert self.resets is not None

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        if self.epochs_per_cycle is not None:
            epochs = self.epochs_per_cycle
        all_scores = []

        for i in range(self.resets):
            # m = deepcopy(self.model)
            # model = self.models[i]
            optim = optimizer([param for name, param in self.model.named_parameters() if param.requires_grad])
            train_scheduler = SnapshotAnnealing(optim,
                                                epochs_per_cycle=epochs,
                                                alpha=None)
            best_model, scores, best_model_scores, losses = train_model(model=self.model, optimizer=optim,
                                                                        epochs=epochs, train_loader=train_dataset,
                                                                        scheduler=train_scheduler,
                                                                        early_stopping=early_stopping,
                                                                        test_loader=test_dataset,
                                                                        eval_loader=eval_dataset,
                                                                        device=self.device)
            self.model.load_state_dict(best_model)
            self.models[i].load_state_dict(best_model)
            all_scores.append(scores)

        return all_scores

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models])
        if reduce:
            outputs = torch.mean(outputs, 0)
        return outputs

    def load(self, path):
        self.device = 'cpu'
        for i in range(self.resets):
            state_dict = torch.load(os.path.join(path, 'model_{}.pt'.format(i)),
                                    map_location=self.device)
            m = deepcopy(self.model)
            m.load_state_dict(state_dict)
            # m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), os.path.join(path, 'model_{}.pt'.format(i)))
