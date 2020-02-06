import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import nn_optimizer
import optimizee


# ZO-SGD
class BaseZOOptimizer(nn_optimizer.NNOptimizer):
    def __init__(self, model, args, lr=25):
        super(BaseZOOptimizer, self).__init__(model, args)
        self.q = args.grad_est_q
        self.lr = lr

    def reset_state(self, keep_states=False, model=None, use_cuda=False, gpu_num=0):
        pass

    def meta_update(self, model, data, target):
        f_x = model(data)
        loss = model.loss(f_x, target)

        flat_grads = torch.zeros_like(model.get_params())
        for _ in range(self.q):
            u = torch.randn_like(model.get_params())  # sampled query direction
            flat_grads += self.GradientEstimate(model, data, target, u) * u
        flat_grads /= self.q

        lr = self.lr / model.get_params_size()
        flat_params = model.get_params()
        delta = - flat_grads * lr
        flat_params = flat_params + delta
        model.set_params(flat_params)

        return None, loss, f_x


# ZO-signSGD
class SignZOOptimizer(nn_optimizer.NNOptimizer):
    def __init__(self, model, args, lr=25):
        super(SignZOOptimizer, self).__init__(model, args)
        self.q = args.grad_est_q
        self.lr = lr

    def reset_state(self, keep_states=False, model=None, use_cuda=False, gpu_num=0):
        pass

    def meta_update(self, model, data, target):
        f_x = model(data)
        loss = model.loss(f_x, target)

        flat_grads = torch.zeros_like(model.get_params())
        for _ in range(self.q):
            u = torch.randn_like(model.get_params())  # sampled query direction
            flat_grads += self.GradientEstimate(model, data, target, u) * u
        flat_grads /= self.q

        lr = self.lr / model.get_params_size()
        flat_params = model.get_params()
        flat_params = flat_params - flat_grads.sign() * lr
        model.set_params(flat_params)

        return None, loss, f_x


# ZO-ADAM
class AdamZOOptimizer(nn_optimizer.NNOptimizer):
    def __init__(self, model, args, lr=25, beta_1=0.9, beta_2=0.996):
        super(AdamZOOptimizer, self).__init__(model, args)
        self.q = args.grad_est_q
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def reset_state(self, keep_states=False, model=None, use_cuda=False, gpu_num=0):
        self.first_moment = torch.zeros_like(model.get_params())
        self.second_moment = torch.zeros_like(model.get_params())
        self.step = 0

    def meta_update(self, model, data, target):
        f_x = model(data)
        loss = model.loss(f_x, target)

        self.step += 1

        flat_grads = torch.zeros_like(model.get_params())
        for _ in range(self.q):
            u = torch.randn_like(model.get_params())  # sampled query direction
            flat_grads += self.GradientEstimate(model, data, target, u) * u
        flat_grads /= self.q

        self.first_moment = self.beta_1 * self.first_moment + (1 - self.beta_1) * flat_grads
        debiased_first_moment = self.first_moment / (1 - self.beta_1 ** self.step)
        self.second_moment = self.beta_2 * self.second_moment + (1 - self.beta_2) * (flat_grads ** 2)
        debiased_second_moment = self.second_moment / (1 - self.beta_2 ** self.step)
        flat_grads = debiased_first_moment / (debiased_second_moment.sqrt() + 1e-8)

        lr = self.lr / model.get_params_size()
        flat_params = model.get_params()
        delta = - flat_grads * lr
        flat_params = flat_params + delta
        model.set_params(flat_params)

        return None, loss, f_x
