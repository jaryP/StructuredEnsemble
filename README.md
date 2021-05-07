## Structured Ensemble

This repository contains a PyTorch implementation of the paper: 

### [Structured Ensembles: an Approach to Reduce the Memory Footprint of Ensemble Methods](https://arxiv.org/abs/2105.02551)
<!--[Structured Ensembles: an Approach to Reduce the Memory Footprint of Ensemble Methods](https://arxiv.org/abs/2105.02551)\ -->
[Jary Pomponi](https://www.semanticscholar.org/author/Jary-Pomponi/1387980523), [Simone Scardapane](https://www.sscardapane.it/), [Aurelio Uncini](http://www.uncini.com/)

### Abstract
In this paper, we propose a novel ensembling technique for deep neural networks, which is able to drastically reduce the
required memory compared to alternative approaches. In particular, we propose to extract multiple sub-networks from a single,
untrained neural network by solving an end-to-end optimization task combining differentiable scaling over the original architecture, with
multiple regularization terms favouring the diversity of the ensemble. Since our proposal aims to detect and extract sub-structures, we
call it Structured Ensemble. On a large experimental evaluation, we show that our method can achieve higher or comparable
accuracy to competing methods while requiring significantly less storage. In addition, we evaluate our ensembles in terms of predictive
calibration and uncertainty, showing they compare favourably with the state-of-the-art. Finally, we draw a link with the continual learning
literature, and we propose a modification of our framework to handle continuous streams of tasks with a sub-linear memory cost. We
compare with a number of alternative strategies to mitigate catastrophic forgetting, highlighting advantages in terms of average
accuracy and memory.

### Main Dependencies
* pytorch==1.7.1
* python=3.8.5
* torchvision==0.8.2
* continual-learning==0.1.6.5
* pyyaml==5.3.1
* tqdm
* dill 
  
### Experiments files
The folder './config/' contains all the yaml files used for the experiments presented in the paper. 

The folders './config/optimizers' and './config/training' contain, respectively, the files which contain the optimizers and the training strategies. 

The folder './config/experiments/classification' contains all the files used for the ensemble experiments, while './config/experiments/classification' contains the ones used in the CL scenarios.

### Training
We have teo training files:

* main.py: to be used only with config files from './config/experiments/classification'
* main_cl.py: to be used only with config files from './config/experiments/cl'

Bot scripts accept any number of training files, which are processed sequentially, and also an optional flag --device [integer|cpu] that can be used to specify the device (otherwise the one present in each config file is used).

Please refer to the yaml files to understand how they can be formatted, and to the methods to understand the parameters that can be used.

If you want to use TinyImagenet you need to download and preprocess it first, using the script 'tinyimagenet_download.sh'.

### Cite

Please cite our work if you find it useful:

```
@misc{pomponi2021structured,
      title={Structured Ensembles: an Approach to Reduce the Memory Footprint of Ensemble Methods}, 
      author={Jary Pomponi and Simone Scardapane and Aurelio Uncini},
      year={2021},
      eprint={2105.02551},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```