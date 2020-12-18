import os
from copy import deepcopy

import torch
from tqdm import tqdm

from methods.base import EnsembleMethod
from utils import train_model


class Naive(EnsembleMethod):
    def __init__(self, model, method_parameters, device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.device = device

        self.model = model
        self.models = torch.nn.ModuleList()

        for i in range(method_parameters['n_ensemble']):
            model = deepcopy(model)
            for name, module in model.named_modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            self.models.append(model)

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
              regularization=None, early_stopping=None,
              **kwargs):

        all_scores = []
        for i in tqdm(range(len(self.models)), desc='Training models'):
            model = self.models[i]
            optim = optimizer([param for name, param in model.named_parameters() if param.requires_grad])
            train_scheduler = scheduler(optim)

            best_model, scores, best_model_scores, losses = train_model(model=model, optimizer=optim,
                                                                epochs=epochs, train_loader=train_dataset,
                                                                scheduler=train_scheduler,
                                                                early_stopping=early_stopping,
                                                                test_loader=test_dataset, eval_loader=eval_dataset,
                                                                device=self.device)

            model.load_state_dict(best_model)
            all_scores.append(scores)

        return all_scores

    def predict_proba(self, x, y, **kwargs):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models])
        outputs = torch.mean(outputs, 0)
        outputs = torch.nn.functional.softmax(outputs, -1)

        return outputs

    def load(self, path):
        for i in range(self.ensemble):
            state_dict = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m = deepcopy(self.model)
            m.load_state_dict(state_dict)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), os.path.join(path, 'model_{}.pt'.format(i)))
