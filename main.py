import os
import sys

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from tqdm import tqdm

from supermask_ensamble import EnsembleSupermaskBeforeTraining
from utils import get_optimizer, get_dataset, get_model, eval_model, EarlyStopping, ensures_path
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

        if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
            print(seed_path, 'loaded')
            continue

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(os.path.join(seed_path, 'info.log'), mode='w'),
                                logging.StreamHandler(sys.stdout)
                            ]
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

        model = get_model(name=trainer['model'], input_size=input_size, output=classes)
        model.to(device)

        method_name = experiment_config.get('method', None).lower()

        scores = []
        best_model = {}

        if to_save:
            torch.save(model.state_dict(), os.path.join(seed_path, 'initial_model.pt'))

        # if method_name is not None and method_name != 'normal':

        method_parameters = dict(experiment_config['method_parameters'])
        models = EnsembleSupermaskBeforeTraining(model=model, train_dataset=train_loader, test_dataset=test_loader,
                                        optimizer=optimizer, eval_dataset=eval_loader,
                                        scheduler=None, regularization=regularization, save_path=None,
                                        device=device,
                                        **method_parameters)
        #
        # if method_name in UNSTRUCTURED_METHODS_MAP:
        #     method = UNSTRUCTURED_METHODS_MAP[method_name]
        #     returns = method(model=model, train_dataset=train_loader, test_dataset=test_loader,
        #                      optimizer=optimizer, eval_dataset=eval_loader,
        #                      scheduler=None, regularization=regularization, save_path=None,
        #                      device=device,
        #                      **method_parameters)
        #
        #     total_param = 0
        #     zero_param = 0
        #
        #     for name, param in model.named_parameters():
        #         nnz = torch.count_nonzero(param.data).item()
        #         zero = param.numel() - nnz
        #
        #         total_param += param.numel()
        #         zero_param += zero
        #
        #         print(name, nnz, zero, zero / param.numel())
        #
        #     print('Total parameters: {}, Non zero parameters: {}, Sparsity: {}'.format(total_param, zero_param,
        #                                                                                zero_param / total_param))
        # elif method_name in STRUCTURED_METHODS_MAP:
        #     # s1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #
        #     s1 = {name: module.weight for name, module in model.named_modules() if hasattr(module, 'weight')}
        #
        #     method = STRUCTURED_METHODS_MAP[method_name]
        #     returns = method(model=model, train_dataset=train_loader, test_dataset=test_loader,
        #                      optimizer=optimizer, eval_dataset=eval_loader,
        #                      scheduler=None, regularization=regularization, save_path=None,
        #                      device=device,
        #                      **method_parameters)
        #
        #     s2 = {name: module.weight for name, module in model.named_modules() if hasattr(module, 'weight')}
        #
        #     logger.info('layers sparsity \n')
        #
        #     for name in s1.keys():
        #         logger.info('Layer: {}. Original parameters: {} ({}), pruned parameters: {} ({})'.
        #                     format(name, s1[name].numel(), s1[name].shape, s2[name].numel(), s2[name].shape))
        #
        #     ss1 = sum(w.numel() for _, w in s1.items())
        #     ss2 = sum(w.numel() for _, w in s2.items())
        #
        #     logger.info('Parameters of original model: {}, parameters of new model: {}. Sparsity: {}'.format(
        #         ss1, ss2, 1 - ss2 / ss1))
        #
        # else:
        #     assert False
        #
        # if isinstance(returns, tuple):
        #     model, hooks = returns
        # else:
        #     model = returns

        optim = optimizer(model.parameters())

        train_scheduler = scheduler(optim)

        model.train()

        for e in tqdm(range(epochs)):
            model.train()
            losses = []
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = nn.functional.cross_entropy(pred, y, reduction='none')
                losses.extend(loss.tolist())
                loss = loss.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()

            if eval_loader is not None:
                eval_scores = eval_model(model, eval_loader, device=device, topk=[1, 5])
            else:
                eval_scores = 0

            losses = sum(losses) / len(losses)

            # if early_stopping is not None:
            #     if early_stopping_value == 'eval':
            #         r = early_stopping.step(eval_scores[1])
            #     else:
            #         r = early_stopping.step(losses)
            #
            #     if r < 0:
            #         break
            #     elif r > 0:
            #         best_model = model.state_dict()

            # train_scores = eval_model(model, train_loader, device=device)
            # test_scores = eval_model(model, test_loader, device=device)
            #
            # logger.info('Epoch {}/{} over. Results:'.format(e + 1, epochs))
            # logger.info('\tTrain: {}'.format(train_scores))
            # logger.info('\tEval: {}'.format(eval_scores))
            # logger.info('\tTest: {}'.format(test_scores))
            #
            # scores.append((train_scores, eval_scores, test_scores))

            if train_scheduler is not None:
                if isinstance(train_scheduler, (StepLR, MultiStepLR)):
                    train_scheduler.step()

        # if to_save:
        #     with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
        #         pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)
        #     torch.save(best_model, os.path.join(seed_path, 'final_model.pt'))
        #     logger.info('\tFinal model and associated results saved.')
        #
        # model.load_state_dict(best_model)

        if eval_loader is not None:
            eval_scores = eval_model(model, eval_loader, device=device, topk=[1, 5])
        else:
            eval_scores = 0

        train_scores = eval_model(model, train_loader, device=device)
        test_scores = eval_model(model, test_loader, device=device)

        logger.info('Results of the best model')
        logger.info('\tTrain: {}'.format(train_scores))
        logger.info('\tEval: {}'.format(eval_scores))
        logger.info('\tTest: {}'.format(test_scores))

        for i, model in enumerate(models):
            optim = optimizer(model.parameters())

            train_scheduler = scheduler(optim)

            for e in tqdm(range(epochs), desc='Model #{}'.format(i)):
                model.train()
                losses = []
                for i, (x, y) in enumerate(train_loader):
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = nn.functional.cross_entropy(pred, y, reduction='none')
                    losses.extend(loss.tolist())
                    loss = loss.mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                if eval_loader is not None:
                    eval_scores = eval_model(model, eval_loader, device=device, topk=[1, 5])
                else:
                    eval_scores = 0

                losses = sum(losses) / len(losses)

                # if early_stopping is not None:
                #     if early_stopping_value == 'eval':
                #         r = early_stopping.step(eval_scores[1])
                #     else:
                #         r = early_stopping.step(losses)
                #
                #     if r < 0:
                #         break
                #     elif r > 0:
                #         best_model = model.state_dict()

                # train_scores = eval_model(model, train_loader, device=device)
                # test_scores = eval_model(model, test_loader, device=device)
                #
                # logger.info('Epoch {}/{} over. Results:'.format(e + 1, epochs))
                # logger.info('\tTrain: {}'.format(train_scores))
                # logger.info('\tEval: {}'.format(eval_scores))
                # logger.info('\tTest: {}'.format(test_scores))
                #
                # scores.append((train_scores, eval_scores, test_scores))
                #
                # if train_scheduler is not None:
                #     if isinstance(train_scheduler, (StepLR, MultiStepLR)):
                #         train_scheduler.step()

                # if to_save:
                #     with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
                #         pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)
                #     torch.save(best_model, os.path.join(seed_path, 'final_model.pt'))
                #     logger.info('\tFinal model and associated results saved.')
                #
                # model.load_state_dict(best_model)

            if eval_loader is not None:
                eval_scores = eval_model(model, eval_loader, device=device, topk=[1, 5])
            else:
                eval_scores = 0

            train_scores = eval_model(model, train_loader, device=device)
            test_scores = eval_model(model, test_loader, device=device)

            logger.info('Results of the best model')
            logger.info('\tTrain: {}'.format(train_scores))
            logger.info('\tEval: {}'.format(eval_scores))
            logger.info('\tTest: {}'.format(test_scores))
