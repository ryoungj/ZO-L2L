# Zeroth-Order Learning to Learn
This repository contains the code for [Learning to Learn by Zeroth-Order Oracle](https://openreview.net/forum?id=ryxz8CVYDH), which extends the learning to learn (L2L) framework to zeroth-order (ZO) optimization.

## Requirements
* Python >= 3.6
* PyTorch >= 1.1.0
* Pillow == 6.1.0
* matplotlib

## Usage
We include the MNIST attack experiment here.

### Train the ZO optimizer
* Train the UpdateRNN
```bash
python main_attack.py --exp_name ZO_attack_mnist --train_task ZOL2L-Attack --gpu_num 0 --train optimizer_attack
```
* Train the QueryRNN (freeze the UpdateRNN)
```bash
python main_attack.py --exp_name VarReduced_ZO_attack_mnist --train_task VarReducedZOL2L-Attack --gpu_num 0 --train optimizer_attack --warm_start_ckpt ./output/ZO_attack_mnist/ckpt_best
```
By default, first-order method is used to train the zeroth-order optimizer (assume the gradient of the optimizee is 
available at training time). You can also add `--use_finite_diff` to use the zeroth-order method (approximate the gradient) 
to train the optimizer, which needs more computation at training time.

### Test the ZO optimizer
* Test the learned ZO optimizer and compare with baseline ZO optimization algorithms
```bash
python main_attack.py --exp_name VarReduced_ZO_attack_mnist --train_task VarReducedZOL2L-Attack --gpu_num 0 --train optimizer_train_optimizee_attack --ckpt_path ckpt_best --save_loss
```
Adding `--save_fig` to the command plots the loss curves of each algorithm for comparison.  
The test settings are listed in the `tests` dict of the corresponding task in `train_task_list.py`, which can be modified easily.

## Customization
New zeroth-order optimization problems and optimizers can be implemented easily on this code base. You can add new 
zeroth-order optimization problems to `optimizee` similar to `optimizee/mnist.py`, and new zeroth-order optimizers to 
`nn_optimizer` similar to `nn_optimizer/zoopt.py`. 

## Cite
```
@inproceedings{
ruan2020learning,
title={Learning to Learn by Zeroth-Order Oracle},
author={Yangjun Ruan and Yuanhao Xiong and Sashank Reddi and Sanjiv Kumar and Cho-Jui Hsieh},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=ryxz8CVYDH}
}
```