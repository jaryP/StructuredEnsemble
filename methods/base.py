import os
from abc import ABC, abstractmethod
from copy import deepcopy

import torch

from utils import train_model


class EnsembleMethod(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super(EnsembleMethod, self).__init__()

    @abstractmethod
    def train_models(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, x, y, reduce):
        raise NotImplementedError

    # @abstractmethod
    def predict_proba(self, x, y, reduce, **kwargs):
        logits = self.predict_logits(x, y, reduce)
        probs = torch.nn.functional.softmax(logits, -1)
        return probs, logits

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, **kwargs):
        pass


class SingleModel(EnsembleMethod):
    def __init__(self, model, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.model = deepcopy(model)
        self.device = device

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):
        optim = optimizer([param for name, param in self.model.named_parameters() if param.requires_grad])
        train_scheduler = scheduler(optim)

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
        return self.model(x)

    def load(self, path):
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
