import os
import pickle
import sys

import dill
import numpy as np
import torch
from continual_learning.eval.metrics.cl import BackwardTransfer, TotalAccuracy
from continual_learning.eval.metrics.classification import Accuracy
from continual_learning.eval.metrics.general import TimeMetric
from continual_learning.scenarios.supervised.task_incremental import MultiTask, \
    GgTrainer
from continual_learning.solvers.multi_task import MultiHeadsSolver
from torch import nn

from utils import get_optimizer, EarlyStopping, ensures_path
from cl_utils import get_cl_dataset, get_cl_model, get_cl_method
import yaml
import logging
import argparse


def solver_fn(input, output):
    return nn.Sequential(*[nn.Linear(input, input // 2),
                           nn.Dropout(0.25),
                           nn.ReLU(),
                           nn.Linear(input // 2, input // 4),
                           nn.Dropout(0.25),
                           nn.ReLU(),
                           nn.Linear(input // 4, output)])


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='Paths of the yaml con figuration files')

parser.add_argument('--device',
                    required=False,
                    default=None,
                    type=int,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

for experiment in args.files:

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

    if args.device is not None:
        device = args.device
    else:
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
    tasks = trainer['tasks']

    for experiment_seed in range(experiments):
        np.random.seed(experiment_seed)
        torch.random.manual_seed(experiment_seed)

        seed_path = os.path.join(save_path, experiment_path,
                                 str(experiment_seed))

        already_present = ensures_path(seed_path)

        hs = [
            logging.StreamHandler(sys.stdout),
            # logging.StreamHandler(sys.stderr)
        ]

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

        if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
            with open(os.path.join(seed_path, 'results.pkl'), 'rb') \
                    as file:
                evaluator = dill.load(file)
        else:
            dataset, input_size, classes = get_cl_dataset(trainer['dataset'],
                                                          model_name=trainer[
                                                              'model'])

            backbone = get_cl_model(trainer['model'], input_size=input_size)
            dataset.create_dev_split(dev_split=0.1)

            mt = MultiTask(dataset=dataset,
                           labels_per_task=len(dataset.labels) // tasks)

            backbone.to(device)

            t0 = mt[0]
            _, img, _ = next(iter(t0.get_iterator(batch_size=1)))

            output_dim = backbone(img.to(device)).shape[1:]
            if len(output_dim) == 1:
                output_dim = output_dim[0]

            solver = MultiHeadsSolver(input_dim=output_dim, topology=solver_fn)

            method_name = experiment_config.get('method', None).lower()

            method = get_cl_method(name=experiment_config.get('method'),
                                   backbone=backbone,
                                   device=device,
                                   parameters=experiment_config.get(
                                       'parameters'))

            method_parameters = dict(
                experiment_config.get('method_parameters', {}))

            t = GgTrainer(backbone=backbone,
                          batch_size=batch_size,
                          optimizer=optimizer,
                          task_epochs=epochs,
                          metrics=[Accuracy(),
                                   BackwardTransfer(),
                                   TotalAccuracy(),
                                   TimeMetric()],
                          device=device,
                          criterion=torch.nn.CrossEntropyLoss(),
                          solver=solver,
                          method=method,
                          tasks=mt)

            t.train_full()

            evaluator = t.evaluator

            if to_save:
                with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
                    dill.dump(evaluator, file,
                              protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(evaluator.task_matrix)
        logger.info(evaluator.cl_results())
        logger.info(evaluator.others_metrics_results())

        # model = get_model(name=trainer['model'], input_size=input_size,
        #                   output=classes)
        #
        # logger.info('Base model parameters: {}'.format(
        #     calculate_trainable_parameters(model)))
        #
        # if method_name is None or method_name == 'normal':
        #     method_name = 'normal'
        #     method = SingleModel(model=model, device=device)
        # elif method_name == 'naive':
        #     method = Naive(model=model, device=device,
        #                    method_parameters=method_parameters)
        # elif method_name == 'supermask':
        #     method = SuperMask(model=model, method_parameters=method_parameters,
        #                        device=device)
        # elif method_name == 'batch_ensemble':
        #     method = BatchEnsemble(model=model,
        #                            method_parameters=method_parameters,
        #                            device=device)
        # elif method_name == 'batch_supermask':
        #     method = ExtremeBatchPruningSuperMask(model=model,
        #                                           method_parameters=method_parameters,
        #                                           device=device)
        #     # method = BatchForwardPruningSuperMask(model=model,
        #     #                                       method_parameters=method_parameters,
        #     #                                       device=device)
        #     # method = BatchPruningSuperMask(model=model, method_parameters=method_parameters, device=device)
        # elif method_name == 'grad_supermask_post':
        #     method = BatchPruningSuperMaskPostTraining(model=model,
        #                                                method_parameters=
        #                                                method_parameters,
        #                                                device=device)
        # # elif method_name == 'reverse_supermask':
        # # method = ReverseSuperMask(model=model, method_parameters=method_parameters, device=device)
        # # elif method_name == 'grad_supermask':
        # # method = GradSuperMask(model=model, method_parameters=method_parameters, device=device)
        # # elif method_name == 'tree_supermask':
        # # method = TreeSuperMask(model=model, method_parameters=method_parameters, device=device)
        # elif method_name == 'mc_dropout':
        #     method = MCDropout(model=model, method_parameters=method_parameters,
        #                        device=device)
        # elif method_name == 'snapshot':
        #     method = Snapshot(model=model, method_parameters=method_parameters,
        #                       device=device)
        # else:
        #     assert False
        #
        # logger.info('Method used: {}'.format(method_name))
        #
        # if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
        #     method.load(os.path.join(seed_path))
        #     with open(os.path.join(seed_path, 'results.pkl'), 'rb') as file:
        #         results = pickle.load(file)
        #     logger.info('Results and models loaded.')
        # else:
        #     results = method.train_models(optimizer=optimizer,
        #                                   train_dataset=train_loader,
        #                                   epochs=epochs,
        #                                   scheduler=scheduler,
        #                                   early_stopping=early_stopping,
        #                                   test_dataset=test_loader,
        #                                   eval_dataset=eval_loader)
        #
        #     if to_save:
        #         method.save(seed_path)
        #         with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
        #             pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        #         logger.info('Results and models saved.')
        #
        # # logger.info('Ensemble score on train: {}'.format(eval_method(method,
        # # dataset=train_loader)[0]))
        # # logger.info('Ensemble score on eval: {}'.format(eval_method(method,
        # # dataset=eval_loader)[0]))
        #
        # logger.info('Ensemble '
        #             'score on test: {}'.format(
        #     eval_method(method, dataset=test_loader)[0]))
        #
        # if to_load and os.path.exists(os.path.join(seed_path, 'fgsm.pkl')):
        #     with open(os.path.join(seed_path, 'fgsm.pkl'), 'rb') as file:
        #         fgsm = pickle.load(file)
        #
        #     logger.info('fgsm results loaded.')
        # else:
        #     fgsm = {}
        #
        #     for e in [0, 0.001, 0.01, 0.02, 0.1, 0.5]:
        #         true, predictions, probs, hs = \
        #             perturbed_predictions(method,
        #                                   test_loader,
        #                                   epsilon=e, device=method.device)
        #
        #         fgsm[e] = {'true': true, 'predictions': predictions,
        #                    'probs': probs, 'entropy': hs}
        #
        #     with open(os.path.join(seed_path, 'fgsm.pkl'), 'wb') as file:
        #         pickle.dump(fgsm, file, protocol=pickle.HIGHEST_PROTOCOL)
        #     logger.info('fgsm results saved.')
        #
        # for e, v in fgsm.items():
        #     true, predictions, probs, hs = v['true'], \
        #                                    v['predictions'], \
        #                                    v['probs'], \
        #                                    v['entropy']
        #
        #     logger.info('FGSM. Epsilon {}'.format(e))
        #     cph = [h for i, h in enumerate(hs) if true[i] == predictions[i]]
        #     wph = [h for i, h in enumerate(hs) if true[i] != predictions[i]]
        #     logger.info('\tCorrectly classified entropy: {} (+-{}) # {} \n\t'
        #                 'Wrongly classified entropy: {} (+-{}) # {}'.format(
        #         np.mean(cph),
        #         np.std(cph),
        #         len(cph),
        #         np.mean(wph),
        #         np.std(wph),
        #         len(wph)))
        #
        # # if trainer['dataset'] in ['cifar10_vgg11', 'cifar100']:
        # #     if to_load and os.path.exists(
        # #             os.path.join(seed_path, 'corrupted.pkl')):
        # #         with open(os.path.join(seed_path, 'corrupted.pkl'),
        # #                   'rb') as file:
        # #             corrupted = pickle.load(file)
        # #
        # #         logger.info('corrupted cifar results loaded.')
        # #
        # #     else:
        # #
        # #         entropy, scores, buc, ece = corrupted_cifar_uncertainty(method,
        # #                                                                 batch_size * 2,
        # #                                                                 dataset=
        # #                                                                 trainer[
        # #                                                                     'dataset'])
        # #
        # #         with open(os.path.join(seed_path, 'corrupted.pkl'),
        # #                   'wb') as file:
        # #             pickle.dump({'entropy': entropy,
        # #                          'scores': scores,
        # #                          'buc': buc,
        # #                          'ece': ece}, file,
        # #                         protocol=pickle.HIGHEST_PROTOCOL)
        # #
        # #         logger.info('corrupted results saved.')
        #
        # params = 0
        # if hasattr(method, 'models'):
        #     models = method.models
        # else:
        #     models = [method.model]
        #
        # for m in models:
        #     params += calculate_trainable_parameters(m)
        #
        # logger.info('Method {} has {} parameters'.format(method_name, params))
        #
        # ece, _, _, _ = ece_score(method, test_loader)
        #
        # logger.info('Ece score: {}'.format(ece))
