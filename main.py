import os
import pickle
import sys

import numpy as np
import torch

from experiments.calibration import ece_score
from eval import eval_method
from experiments.corrupted_cifar import corrupted_cifar_uncertainty
from experiments.fgsm import perturbed_predictions
from methods import SingleModel, Naive, SuperMask
from methods.batch_ensemble.batch_ensemble import BatchEnsemble
from methods.dropout.dropout import MCDropout
from methods.snapshot.snapshot import Snapshot
from methods.supermask.supermask import ExtremeBatchPruningSuperMask
from methods.supermask.supermask_after_training import \
    BatchPruningSuperMaskPostTraining
from methods.supermask.supermask_training import BatchForwardPruningSuperMask
from utils import get_optimizer, get_dataset, get_model, EarlyStopping, \
    ensures_path, calculate_trainable_parameters
import yaml
import logging

# TODO: implemetare attacco

# TODO: implementare linear layers BE

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
        early_stopping = EarlyStopping(
            tolerance=early_stopping_dict['tolerance'],
            min=early_stopping_value == 'loss')
    else:
        early_stopping = None
        early_stopping_value = None

    batch_size = trainer['batch_size']
    epochs = trainer['epochs']
    experiments = trainer.get('experiments', 1)

    train, test, input_size, classes = get_dataset(trainer['dataset'],
                                                   model_name=trainer['model'])

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True, num_workers=4)
    eval_loader = None

    for experiment_seed in range(experiments):
        np.random.seed(experiment_seed)
        torch.random.manual_seed(experiment_seed)

        seed_path = os.path.join(save_path, experiment_path,
                                 str(experiment_seed))

        already_present = ensures_path(seed_path)

        hs = [logging.StreamHandler(sys.stdout)]
        hs.append(
            logging.FileHandler(os.path.join(seed_path, 'info.log'), mode='w'))

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=hs
                            )

        logger = logging.getLogger(__name__)

        config_info = experiment_config.copy()
        config_info.update({'optimizer': optimizer_config, 'trainer': trainer})

        logger.info('Config file \n{}'.format(
            yaml.dump(config_info, allow_unicode=True,
                      default_flow_style=False)))

        logger.info('Experiment {}/{}'.format(experiment_seed + 1, experiments))

        if eval_percentage is not None and eval_percentage > 0:
            assert eval_percentage < 1
            train_len = len(train)
            eval_len = int(train_len * eval_percentage)
            train_len = train_len - eval_len

            train, eval = torch.utils.data.random_split(train,
                                                        [train_len, eval_len])
            train_loader = torch.utils.data.DataLoader(dataset=train,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            eval_loader = torch.utils.data.DataLoader(dataset=eval,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            logger.info('Train dataset size: {}'.format(len(train)))
            logger.info('Test dataset size: {}'.format(len(test)))
            logger.info(
                'Eval dataset created, having size: {}'.format(len(eval)))

        method_name = experiment_config.get('method', None).lower()

        method_parameters = dict(experiment_config.get('method_parameters', {}))

        model = get_model(name=trainer['model'], input_size=input_size,
                          output=classes)

        logger.info('Base model parameters: {}'.format(
            calculate_trainable_parameters(model)))

        if method_name is None or method_name == 'normal':
            method_name = 'normal'
            method = SingleModel(model=model, device=device)
        elif method_name == 'naive':
            method = Naive(model=model, device=device,
                           method_parameters=method_parameters)
        elif method_name == 'supermask':
            method = SuperMask(model=model, method_parameters=method_parameters,
                               device=device)
        elif method_name == 'batch_ensemble':
            method = BatchEnsemble(model=model,
                                   method_parameters=method_parameters,
                                   device=device)
        elif method_name == 'batch_supermask':
            # method = ExtremeBatchPruningSuperMask(model=model,
            #                                       method_parameters=method_parameters,
            #                                       device=device)
            method = BatchForwardPruningSuperMask(model=model,
                                                  method_parameters=method_parameters,
                                                  device=device)
            # method = BatchPruningSuperMask(model=model, method_parameters=method_parameters, device=device)
        elif method_name == 'grad_supermask_post':
            method = BatchPruningSuperMaskPostTraining(model=model,
                                                       method_parameters=
                                                       method_parameters,
                                                       device=device)
        # elif method_name == 'reverse_supermask':
        # method = ReverseSuperMask(model=model, method_parameters=method_parameters, device=device)
        # elif method_name == 'grad_supermask':
        # method = GradSuperMask(model=model, method_parameters=method_parameters, device=device)
        # elif method_name == 'tree_supermask':
        # method = TreeSuperMask(model=model, method_parameters=method_parameters, device=device)
        elif method_name == 'mc_dropout':
            method = MCDropout(model=model, method_parameters=method_parameters,
                               device=device)
        elif method_name == 'snapshot':
            method = Snapshot(model=model, method_parameters=method_parameters,
                              device=device)
        else:
            assert False

        logger.info('Method used: {}'.format(method_name))

        if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
            method.load(os.path.join(seed_path))
            with open(os.path.join(seed_path, 'results.pkl'), 'rb') as file:
                results = pickle.load(file)
            logger.info('Results and models loaded.')
        else:
            results = method.train_models(optimizer=optimizer,
                                          train_dataset=train_loader,
                                          epochs=epochs,
                                          scheduler=scheduler,
                                          early_stopping=early_stopping,
                                          test_dataset=test_loader,
                                          eval_dataset=eval_loader)

            if to_save:
                method.save(seed_path)
                with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
                    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info('Results and models saved.')

        # logger.info('Ensemble score on train: {}'.format(eval_method(method,
        # dataset=train_loader)[0]))
        # logger.info('Ensemble score on eval: {}'.format(eval_method(method,
        # dataset=eval_loader)[0]))

        logger.info('Ensemble '
                    'score on test: {}'.format(
            eval_method(method, dataset=test_loader)[0]))

        if to_load and os.path.exists(os.path.join(seed_path, 'fgsm.pkl')):
            with open(os.path.join(seed_path, 'fgsm.pkl'), 'rb') as file:
                fgsm = pickle.load(file)

            logger.info('fgsm results loaded.')
        else:
            fgsm = {}

            for e in [0, 0.001, 0.01, 0.02, 0.1]:
                true, predictions, probs, hs = \
                    perturbed_predictions(method,
                                          test_loader,
                                          epsilon=e, device=device)
                fgsm[e] = {'true': true, 'predictions': predictions,
                           'probs': probs, 'entropy': hs}

            with open(os.path.join(seed_path, 'fgsm.pkl'), 'wb') as file:
                pickle.dump(fgsm, file, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('fgsm results saved.')

        if trainer['dataset'] in ['cifar10', 'cifar100']:
            if to_load and os.path.exists(os.path.join(seed_path, 'corrupted.pkl')):
                with open(os.path.join(seed_path, 'corrupted.pkl'), 'rb') as file:
                    fgsm = pickle.load(file)

                logger.info('corrupted cifar results loaded.')

            else:

                entropy, scores, buc, ece = corrupted_cifar_uncertainty(method,
                                                                        batch_size*2,
                                                                        dataset=
                                                                        trainer[
                                                                            'dataset'])

                with open(os.path.join(seed_path, 'corrupted.pkl'), 'wb') as file:
                    pickle.dump({'entropy': entropy,
                                 'scores': scores,
                                 'buc': buc,
                                 'ece': ece}, file,
                                protocol=pickle.HIGHEST_PROTOCOL)

                logger.info('corrupted results saved.')

        params = 0
        if hasattr(method, 'models'):
            models = method.models
        else:
            models = [method.model]

        for m in models:
            params += calculate_trainable_parameters(m)

        logger.info('Method {} has {} parameters'.format(method_name, params))

        ece, _, _, _ = ece_score(method, test_loader)

        logger.info('Ece score: {}'.format(ece))
