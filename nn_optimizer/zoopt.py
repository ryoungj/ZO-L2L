import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os

import nn_optimizer
import optimizee


# ZO optimizer (UpdateRNN only)
class ZOOptimizer(nn_optimizer.NNOptimizer):

    def __init__(self, model, args, num_layers=1, input_dim=1, hidden_size=10):
        super(ZOOptimizer, self).__init__(model, args)

        self.update_rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False)
        self.outputer = nn.Linear(hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.q = args.grad_est_q

    def reset_state(self, keep_states=False, model=None, use_cuda=False, gpu_num=0):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            self.h0 = Variable(self.h0.data)
            self.c0 = Variable(self.c0.data)
        else:
            def initialize_rnn_hidden_state(dim_sum, n_layers, n_params):
                h0 = Variable(torch.zeros(n_layers, n_params, dim_sum), requires_grad=True)
                if use_cuda:
                    h0.data = h0.data.cuda(gpu_num)
                return h0

            self.h0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.step = 0

    def forward(self, x):
        output1, (hn1, cn1) = self.update_rnn(x, (self.h0, self.c0))
        self.h0 = hn1
        self.c0 = cn1
        o1 = self.outputer(output1)
        return o1.squeeze()

    def meta_update(self, model, data, target):
        # compute the zeroth-order gradient estimate of the model
        f_x = model(data)
        loss = model.loss(f_x, target)

        self.step += 1

        flat_grads = torch.zeros_like(model.get_params())
        for _ in range(self.q):
            u = torch.randn_like(model.get_params())  # sampled query direction
            flat_grads += self.GradientEstimate(model, data, target, u) * u
        flat_grads /= self.q

        flat_params = self.meta_model.get_flat_params()
        inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))

        # Meta update itself
        delta = self(inputs)
        flat_params = flat_params + delta

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model)
        return self.meta_model.model, loss, f_x


# ZO optimizer (both UpdateRNN and QueryRNN)
class VarReducedZOOptimizer(nn_optimizer.NNOptimizer):

    def __init__(self, model, args, num_layers=1, input_dim=1, hidden_size=10, ckpt_path=""):
        super(VarReducedZOOptimizer, self).__init__(model, args)

        self.update_rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bias=False)
        self.outputer = nn.Linear(hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)

        self.query_u_rnn = nn.LSTM(input_dim * 2, hidden_size, num_layers, batch_first=True, bias=False)
        self.query_u_outputer = nn.Linear(hidden_size, input_dim, bias=False)
        self.query_u_outputer.weight.data.mul_(0.1)

        self.reg_loss = Variable(torch.zeros(()), requires_grad=True)
        self.reg_lambda = 0.005
        print(">>>reg", self.reg_lambda)

        self.normalize = True

        self.last_grad_est = Variable(torch.zeros(self.meta_model.get_flat_params().size(0)), requires_grad=False)
        self.last_param_update = Variable(torch.zeros(self.meta_model.get_flat_params().size(0)), requires_grad=False)
        if args.cuda:
            self.last_grad_est = self.last_grad_est.cuda(args.gpu_num)
            self.last_param_update = self.last_param_update.cuda(args.gpu_num)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.q = args.grad_est_q

        if ckpt_path != "":
            self.warm_start(ckpt_path)

    def warm_start(self, ckpt_path, freeze=True):
        assert os.path.isfile(ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        model_dict = self.state_dict()
        model_dict.update(ckpt_dict['state_dict'])
        self.load_state_dict(model_dict)
        msg = "Warm start from '{}'.".format(ckpt_path)
        if freeze:
            msg = msg + " Freeze parameters."
            for name, param in self.named_parameters():
                if name in ckpt_dict['state_dict'].keys():
                    param.requires_grad = False
        else:
            msg = msg + " Not freeze parameters."
        print(msg)

    def reset_state(self, keep_states=False, model=None, use_cuda=False, gpu_num=0):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            self.h0 = Variable(self.h0.data)
            self.h1 = Variable(self.h1.data)
            self.c0 = Variable(self.c0.data)
            self.c1 = Variable(self.c1.data)
        else:
            def initialize_rnn_hidden_state(dim_sum, n_layers, n_params):
                h0 = Variable(torch.zeros(n_layers, n_params, dim_sum), requires_grad=True)
                if use_cuda:
                    h0.data = h0.data.cuda(gpu_num)
                return h0

            self.h0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.h1 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.c1 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers,
                                                  self.meta_model.get_flat_params().size(0))
            self.step = 0

    def forward(self, x):
        output1, (hn1, cn1) = self.update_rnn(x, (self.h0, self.c0))
        self.h0 = hn1
        self.c0 = cn1
        o1 = self.outputer(output1)
        return o1.squeeze()

    def meta_update(self, model, data, target, pred_update=True, pred_query=True, guided=False, base_lr=4):
        # compute the zeroth-order gradient estimate of the model
        f_x = model(data)
        loss = model.loss(f_x, target)

        self.step += 1

        input2 = torch.cat((self.last_grad_est.unsqueeze(1), self.last_param_update.unsqueeze(1)), dim=1)
        input2.unsqueeze_(1)
        output2, (hn2, cn2) = self.query_u_rnn(input2, (self.h1, self.c1))
        self.h1 = hn2
        self.c1 = cn2
        o2 = self.query_u_outputer(output2)
        o2 = o2.squeeze(1)

        self.std = o2[:, 0]
        self.mean = torch.zeros_like(self.std)
        self.reg_loss = (torch.sum(self.mean ** 2) + torch.sum(self.std ** 2)) * self.reg_lambda
        self.std = self.std + 1.0

        if self.normalize:
            self.std = self.std / self.std.norm() * torch.ones_like(self.std).norm()

        flat_grads = torch.zeros_like(model.get_params())
        if not guided:  # QueryRNN to propose sampling covariance
            pred_ratio = 0.5
            if self.training:
                pred = True
            else:
                pred = np.random.rand() < pred_ratio

            for i in range(self.q):
                u = torch.randn_like(model.get_params())  # sampled query direction
                if pred_query and pred:
                    u = u * self.std.abs() + self.mean
                flat_grads += self.GradientEstimate(model, data, target, u) * u
        else:  # use Guided-ES to propose sampling covariance
            k = 2
            n = model.get_params().size(0)
            alpha = 0.5
            q, _ = torch.qr(torch.cat((self.last_grad_est.unsqueeze(1), self.last_param_update.unsqueeze(1)), dim=1))

            for _ in range(self.q):
                u = torch.randn_like(model.get_params())  # sampled query direction
                norm_prev = u.norm()
                u_k = u.new_tensor(torch.randn((k,)))

                u = ((alpha) ** 0.5) * u + (((1. - alpha)) ** 0.5) * torch.matmul(q, u_k)
                norm_cur = u.norm()
                u = u / norm_cur * norm_prev
                flat_grads += self.GradientEstimate(model, data, target, u) * u
        flat_grads /= self.q

        flat_params = self.meta_model.get_flat_params()
        inputs = flat_grads.view(-1, 1).unsqueeze(1)

        # Meta update itself
        if pred_update:
            delta = self(inputs)
        else:
            lr = base_lr / model.get_params_size()
            delta = - flat_grads * lr

        flat_params = flat_params + delta

        self.last_grad_est = Variable(flat_grads.data, requires_grad=False)
        self.last_param_update = Variable(delta.data, requires_grad=False)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model)
        return self.meta_model.model, loss, f_x


