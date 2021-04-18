# Structured Ensemble

This repository contains a PyTorch implementation of the paper: 

### Many Tickets Are Better Than One: an Efficient Approach to Compute Deep Ensembles
<!--[Many Tickets Are Better Than One: an Efficient Approach to Compute Deep Ensembles]()\ -->
[Jary Pomponi](https://www.semanticscholar.org/author/Jary-Pomponi/1387980523), [Simone Scardapane](http://ispac.diet.uniroma1.it/scardapane/), [Aurelio Uncini](http://www.uncini.com/)

### Abstract
Deep ensembles of neural networks are capable of achieving better performances than standard neural networks. Also, these models are better calibrated and can be used to detect out-of-distribution samples, due the ability of assigning aun uncertainty to a prediction. 
These model are not quite used, since the memory footprint is prohibitive. For this reason we propose a new approach, called Structured Ensemble. It is capable of extracting multiple sub-structure from a single untrained model. The structure are used to build a smaller ensemble, which achieve good performances. 

Also, we used our approach also to solve multiple Continual Learning (Cl) scenarios. 
### Main Dependencies
* pytorch==1.7.1
* python=3.8.5
* torchvision==0.8.2
* continual-learning==0.1.6.5
* pyyaml==5.3.1

The complete list can be found in the environment file env.yml. 

### Experiments files
The folder './config/' contains all the yaml files used for the experiments presented in the paper. 

The folders './config/optimizers' and './config/training' contain, respectively, the files which contain the optimizers and the training strategies. 

The folder './config/experiments/classification' contains all the files used for the ensemble experiments, while './config/experiments/classification' contains the ones used in the CL scenarios.

### Training
We have teo training files:

* main.py: to be used only with config files from './config/experiments/classification'
* main_cl.py: to be used only with config files from './config/experiments/cl'

Bot scripts accept any number of training files, which are processed sequentially, and also an optional flag --device [integer|cpu] that can be used to specify the device (otherwise the one present in the config files are used).

Please refer to the yaml files to understand how they can be formatted, and to the methods to understand the parameters that can be used. 

### Cite

Please cite our work if you find it useful:

```
@article{
}
```