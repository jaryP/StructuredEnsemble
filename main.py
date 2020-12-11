import os
import sys
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from eval import eval_models
from methods.supermask.supermask_ensamble import EnsembleSupermaskBeforeTraining
from utils import get_optimizer, get_dataset, get_model, EarlyStopping, ensures_path, train_model
import yaml
import logging

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

        # if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
        #     print(seed_path, 'loaded')
        #     continue

        hs = [logging.StreamHandler(sys.stdout)]
        # if to_save:
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

        if method_name is None:
            method_name = 'normal'

        if method_name == 'normal':
            models = [get_model(name=trainer['model'], input_size=input_size, output=classes)]

        elif method_name == 'naive':
            method_parameters = dict(experiment_config['method_parameters'])
            n_ensemble = method_parameters['n_ensemble']

            models = [get_model(name=trainer['model'], input_size=input_size, output=classes) for i in
                      range(n_ensemble)]

        elif method_name == 'supermask':
            model = get_model(name=trainer['model'], input_size=input_size, output=classes)
            method_parameters = dict(experiment_config['method_parameters'])
            models = EnsembleSupermaskBeforeTraining(model=model, train_dataset=train_loader, test_dataset=test_loader,
                                                     optimizer=optimizer, eval_dataset=eval_loader,
                                                     scheduler=None, regularization=regularization, save_path=None,
                                                     device=device,
                                                     **method_parameters)
        else:
            assert False

        # model = get_model(name=trainer['model'], input_size=input_size, output=classes)
        #
        # m = deepcopy(model).to(device)
        # optim = optimizer(m.parameters())
        # train_scheduler = scheduler(optim)
        #
        # best_model, scores, best_model_scores = train_model(model=m, optimizer=optim,
        #                                                     train_loader=train_loader,
        #                                                     epochs=epochs,
        #                                                     scheduler=train_scheduler,
        #                                                     early_stopping=None,
        #                                                     test_loader=test_loader, eval_loader=eval_loader,
        #                                                     device=device)
        #
        # print('Best model scores:\nTrain {}, Eval: {}, Test: {}'.format(best_model_scores[0], best_model_scores[1],
        #                                                                 best_model_scores[2]))
        #
        # method_parameters = dict(experiment_config['method_parameters'])
        # models = EnsembleSupermaskBeforeTraining(model=model, train_dataset=train_loader, test_dataset=test_loader,
        #                                          optimizer=optimizer, eval_dataset=eval_loader,
        #                                          scheduler=None, regularization=regularization, save_path=None,
        #                                          device=device,
        #                                          **method_parameters)
        #
        # complete_model_params = sum(module.weight.numel() for name, module in model.named_modules()
        #                             if hasattr(module, 'weight'))

        tot_params = 0
        for i in tqdm(range(len(models)), desc='Training models'):
            model = models[i]
            model_params = sum(module.weight.numel() for name, module in model.named_modules()
                               if hasattr(module, 'weight'))
            tot_params += model_params
            # print(complete_model_params, model_params, complete_model_params / model_params,
            #       1 - (model_params / complete_model_params))

            early_stopping.reset()
            optim = optimizer(model.parameters())

            train_scheduler = scheduler(optim)

            best_model, scores, best_model_scores = train_model(model=model, optimizer=optim, train_loader=train_loader,
                                                                epochs=epochs,
                                                                scheduler=train_scheduler,
                                                                early_stopping=early_stopping,
                                                                test_loader=test_loader, eval_loader=eval_loader,
                                                                device=device)

            model.load_state_dict(best_model)
            logger.info('Best model #{} scores:\nTrain {}, Eval: {}, Test: {}'.format(i, best_model_scores[0],
                                                                                      best_model_scores[1],
                                                                                      best_model_scores[2]))

            # past_models = models[:i]
            # print(len(models[:i]))
            # model = [m] +

        logger.info('Ensemble score on train: {}'.format(eval_models(models, dataset=train_loader, device=device)))
        logger.info('Ensemble score on eval: {}'.format(eval_models(models, dataset=eval_loader, device=device)))
        logger.info('Ensemble score on test: {}'.format(eval_models(models, dataset=test_loader, device=device)))
        logger.info('Total number of parameters: {}'.format(tot_params))
