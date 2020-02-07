'''The model and optimizee used when training.'''
from torch import optim

import nn_optimizer
import optimizee

# Only `MNIST attack` task support yet
tasks = {
    # train ZO optimizer (UpdateRNN only) for MNIST attack
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
            'optimizee': optimizee.mnist.MnistAttack,
            'test_indexes': list(range(1, 11)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.ZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
        }
    },
    # train ZO optimizer (both UpdateRNN and QueryRNN) for MNIST attack
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
        'tests': {
            'optimizee': optimizee.mnist.MnistAttack,
            'test_indexes': list(range(1, 11)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
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
