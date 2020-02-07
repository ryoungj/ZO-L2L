import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Subset

import optimizee

import os
import numpy as np
import random
import time


class MnistModel(optimizee.Optimizee):
    def __init__(self):
        super(MnistModel, self).__init__()

    @staticmethod
    def dataset_loader(data_dir, batch_size, test_batch_size):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader

    def loss(self, fx, tgt):
        loss = F.nll_loss(fx, tgt)
        return loss


class MnistLinearModel(MnistModel):
    '''A MLP on dataset MNIST.'''

    def __init__(self):
        super(MnistLinearModel, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)


class MnistConvModel(MnistModel):
    def __init__(self):
        super(MnistConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MnistAttack(optimizee.Optimizee):
    def __init__(self, attack_model, batch_size=1, channel=1, width=28, height=28, c=0.1, gap=0.0,
                 loss_type="l1", initial_noise=True):
        super(MnistAttack, self).__init__()

        if not initial_noise:
            self.weight = torch.nn.Parameter(torch.zeros((batch_size, channel, width, height)))
        else:
            torch.random.manual_seed(1234)
            self.weight = torch.nn.Parameter(1e-4 * torch.normal(torch.zeros((batch_size, channel, width, height)),
                                                                 torch.ones((batch_size, channel, width, height))))
            torch.random.manual_seed(time.time())
        self.c = c  # regularization parameter c trades off adversarial success and L2 distortion
        self.gap = gap  # confidence parameter that guarantees a constant gap

        self.attack_model = attack_model

        self.bs = batch_size

        self.loss_type = loss_type

    @staticmethod
    def dataset_loader(data_dir, batch_size, test_batch_size, train_num=100, test_num=100):
        path = os.path.join(data_dir, "mnist_correct/label_correct_index.npy")
        label_correct_indices = list(np.load(path))
        random.seed(1234)
        random.shuffle(label_correct_indices)
        train_indices = label_correct_indices[:train_num]
        test_indices = label_correct_indices[5000:5000 + test_num]

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices), drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            Subset(datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), test_indices),
            batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def data_denormalize(data):
        return data * 0.3081 + 0.1307

    def forward(self, x):
        x_ = x
        x = x * 0.3081 + 0.1307  # [0, 1]

        perturb = self.weight

        fx = x + perturb
        fx.clamp_(0.0, 1.0)

        fx = (fx - 0.1307) / 0.3081
        return fx, x_

    def loss(self, fx, tgt, return_tuple=False):
        assert isinstance(fx, tuple)
        x_attack, x = fx

        if self.loss_type == "l1":
            loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum()
        elif self.loss_type == "l2":
            loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum()

        pred_scores = self.attack_model.model(x_attack)  # log likelyhood: (B, 10)
        tgt_onehot = F.one_hot(tgt, num_classes=10).double()  # (B, 10)

        correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=1)
        assert torch.equal(correct_indices, tgt), (
            correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
        max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=1)
        loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
        loss_attack = loss_attack.mean()

        if not return_tuple:
            return (loss_attack + self.c * loss_distort) * self.bs
        else:
            return (loss_attack, self.c * loss_distort)

    def nondiff_loss(self, weight, x, tgt, batch_weight=False):
        if not batch_weight:
            x_ = x
            x = x * 0.3081 + 0.1307  # [0, 1]

            fx = x + weight
            fx.clamp_(0.0, 1.0)

            fx = (fx - 0.1307) / 0.3081

            x_attack, x = fx, x_

            if self.loss_type == "l1":
                loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum()
            elif self.loss_type == "l2":
                loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum()

            pred_scores = self.attack_model.model(x_attack)  # log likelyhood: (B, 10)
            tgt_onehot = F.one_hot(tgt, num_classes=10).double()  # (B, 10)

            correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=1)
            assert torch.equal(correct_indices, tgt), (
                correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
            max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=1)
            loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
            loss_attack = loss_attack.mean()

            return (loss_attack + self.c * loss_distort) * self.bs
        else:
            x_ = x.unsqueeze(0)  # (1, B, *)
            x = x * 0.3081 + 0.1307  # [0, 1]

            fx = x + weight  # (B_weight, B, *)
            fx.clamp_(0.0, 1.0)

            fx = (fx - 0.1307) / 0.3081

            x_attack, x = fx, x_
            x_attack_shape = x_attack.size()

            if self.loss_type == "l1":
                loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum(
                    dim=[1, 2, 3, 4])
            elif self.loss_type == "l2":
                loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum(dim=[1, 2, 3, 4])

            pred_scores = self.attack_model.model(
                x_attack.view(-1, x_attack_shape[2], x_attack_shape[3], x_attack_shape[4])).view(x_attack_shape[0],
                                                                                                 x_attack_shape[1],
                                                                                                 10)  # log likelyhood: (B_weight, B, 10)
            tgt_onehot = F.one_hot(tgt, num_classes=10).double().unsqueeze(1)  # (1, B, 10)

            correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=2)  # (B_weight, B)
            assert torch.equal(correct_indices, tgt.expand_as(correct_indices)), (
                correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
            max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=2)  # (B_weight, B)
            loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
            loss_attack = loss_attack.mean(dim=1)

            return (loss_attack + self.c * loss_distort) * self.bs
