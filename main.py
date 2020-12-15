import os
import pickle
import sys

import numpy as np
import torch

from calibration import ece_score
from eval import eval_method
from methods import SingleModel, Naive
from methods.batch_ensemble.batch_ensemble import BatchEnsemble
from methods.supermask.supermask import SuperMask, BatchSuperMask, ReverseSuperMask
from utils import get_optimizer, get_dataset, get_model, EarlyStopping, ensures_path
import yaml
import logging

#TODO: implementare stampa dei risultati nel file di log
#TODO: implementare ECE sscore (calibrazione)
#TODO: implemetare attacco
#TODO: implementare conteggio parametri

#TODO: implementare linear layers BE

for experiment in sys.argv[1:]:

    with open(experiment, 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    to_save = experiment_config.get('save', True)
    to_load = experiment_config.get('load', True)

    save_path = experiment_config['save_path']
    experiment_path = str(experiment_config['name'])

    ensures_path(save_path)

    with open(experiment_config['optimizer'], 'r') as opt:
        optimizer_config = yaml.safe_load(opt)
        optimizer_config = dict(optimizer_config)

    optimizer, regularization, scheduler = get_optimizer(**optimizer_config)

    device = experiment_config.get('device', 'cpu')
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    device = torch.device(device)

    with open(experiment_config['trainer'], 'r') as opt:
        trainer = yaml.safe_load(opt)
        trainer = dict(trainer)
    eval_percentage = trainer.get('eval', None)

    if 'early_stopping' in trainer:
        early_stopping_dict = dict(trainer['early_stopping'])
        early_stopping_value = early_stopping_dict.get('value', 'loss')
        assert early_stopping_value in ['eval', 'loss']
        if early_stopping_value == 'eval':
            assert eval_percentage is not None and eval_percentage > 0
        early_stopping = EarlyStopping(tolerance=early_stopping_dict['tolerance'],
                                       min=early_stopping_value == 'loss')
    else:
        early_stopping = None
        early_stopping_value = None

    batch_size = trainer['batch_size']
    epochs = trainer['epochs']
    experiments = trainer.get('experiments', 1)

    train, test, input_size, classes = get_dataset(trainer['dataset'])

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    eval_loader = None

    for experiment_seed in range(experiments):
        np.random.seed(experiment_seed)
        torch.random.manual_seed(experiment_seed)

        seed_path = os.path.join(save_path, experiment_path, str(experiment_seed))

        already_present = ensures_path(seed_path)

        hs = [logging.StreamHandler(sys.stdout)]
        hs.append(logging.FileHandler(os.path.join(seed_path, 'info.log'), mode='w'))

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=hs
                            )

        logger = logging.getLogger(__name__)

        config_info = experiment_config.copy()
        config_info.update({'optimizer': optimizer_config, 'trainer': trainer})

        logger.info('Config file \n{}'.format(yaml.dump(config_info, allow_unicode=True, default_flow_style=False)))

        logger.info('Experiment {}/{}'.format(experiment_seed, experiments))

        if eval_percentage is not None and eval_percentage > 0:
            assert eval_percentage < 1
            train_len = len(train)
            eval_len = int(train_len * eval_percentage)
            train_len = train_len - eval_len

            train, eval = torch.utils.data.random_split(train, [train_len, eval_len])
            train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            eval_loader = torch.utils.data.DataLoader(dataset=eval, batch_size=batch_size, shuffle=False)

            logger.info('Train dataset size: {}'.format(len(train)))
            logger.info('Test dataset size: {}'.format(len(test)))
            logger.info('Eval dataset created, having size: {}'.format(len(eval)))

        method_name = experiment_config.get('method', None).lower()

        method_parameters = dict(experiment_config.get('method_parameters', {}))

        model = get_model(name=trainer['model'], input_size=input_size, output=classes)

        if method_name is None or method_name == 'normal':
            method_name = 'normal'
            method = SingleModel(model=model, device=device)
        elif method_name == 'naive':
            method = Naive(model=model, device=device, method_parameters=method_parameters)
        elif method_name == 'supermask':
            method = SuperMask(model=model, method_parameters=method_parameters, device=device)
        elif method_name == 'batch_ensemble':
            method = BatchEnsemble(model=model, method_parameters=method_parameters, device=device)
        elif method_name == 'batch_supermask':
            method = BatchSuperMask(model=model, method_parameters=method_parameters, device=device)
        elif method_name == 'reverse_supermask':
            method = ReverseSuperMask(model=model, method_parameters=method_parameters, device=device)
        else:
            assert False

        logger.info('Method used: {}'.format(method_name))

        if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
            method.load(os.path.join(seed_path))
            with open(os.path.join(seed_path, 'results.pkl'), 'rb') as file:
                results = pickle.load(file)
            logger.info('Results and models loaded.')
        else:
            results = method.train(optimizer=optimizer, train_dataset=train_loader, epochs=epochs,
                                   scheduler=scheduler, early_stopping=early_stopping,
                                   test_dataset=test_loader, eval_dataset=eval_loader)

            if to_save:
                method.save(seed_path)
                with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
                    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info('Results and models saved.')

        logger.info('Ensemble score on train: {}'.format(eval_method(method, dataset=train_loader)))
        logger.info('Ensemble score on eval: {}'.format(eval_method(method, dataset=eval_loader)))
        logger.info('Ensemble score on test: {}'.format(eval_method(method, dataset=test_loader)))

        params = 0
        if hasattr(method, 'models'):
            models = method.models
        else:
            models = [method.model]

        for i in models:
            for n, p in i.named_parameters():
                if p.requires_grad:
                    params += p.numel()

        logger.info('Method {} has {} parameters'.format(method_name, params))

        ece, _, _, _ = ece_score(method, test_loader)

        logger.info('Ece score: {}'.format(ece))
