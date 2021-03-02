import numpy as np


class SnapshotAnnealing:
    def __init__(self, optimizer, alpha, epochs_per_cycle, verbose=False):
        self.epochs_per_cycle = epochs_per_cycle
        self.alpha = alpha
        self.current_epoch = -1
        self.optimizer = optimizer
        self.verbose = verbose

        for param_group in optimizer.param_groups:
            self.alpha = param_group['lr']

    def step(self):
        self.current_epoch += 1
        lr = self.alpha * (np.cos(np.pi * self.current_epoch /
                                  self.epochs_per_cycle) + 1) / 2
        if self.verbose:
            print(self.alpha, lr)
        self.optimizer.state_dict()['param_groups'][0]['lr'] = lr
