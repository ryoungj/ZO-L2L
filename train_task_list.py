'''The model and optimizee used when training.'''
from torch import optim

import nn_optimizer
import optimizee

tasks = {
    'ZOL2L-Attack': {
        'nn_optimizer': nn_optimizer.zoopt.ZOOptimizer,
        'optimizee': optimizee.mnist.MnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 1e-3,
        "max_epoch": 20,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.MnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/mnist_cnn.pt",
        'tests': {
            'test_idx': 1,
            'test_num': 1,
            'n_steps': 200,
            'optimizee': optimizee.mnist.MnistAttack,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.ZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
        }
    },
    'VarReducedZOL2L-Attack': {
        'nn_optimizer': nn_optimizer.zoopt.VarReducedZOOptimizer,
        'optimizee': optimizee.mnist.MnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 0.005,
        "max_epoch": 40,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.MnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/mnist_cnn.pt",
        'warm_start_ckpt': "./output/ZO_attack_mnist_2/ckpt_best",
        'tests': {
            'test_idx': 1,
            'n_steps': 200,
            'test_num': 10,
            'optimizee': optimizee.mnist.MnistAttack,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.VarReducedZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'sign_opt': nn_optimizer.basezoopt.SignZOOptimizer,
            'sign_lr': 8,
            'adam_opt': nn_optimizer.basezoopt.AdamZOOptimizer,
            'adam_lr': 8,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.996,
            # 'nn_opt_no_query': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_no_update': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_guided': nn_optimizer.zoopt.VarReducedZOOptimizer,
        }
    },
}
