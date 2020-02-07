import argparse
import os
import sys
import copy
import numpy as np
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

import train_task_list
import optimizee
import nn_optimizer


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_optimizer_attack(args):
    assert "Attack" in args.train_task
    task = train_task_list.tasks[args.train_task]

    print("Training ZO optimizer...\nOptimizer: {}. Optimizee: {}".format(task["nn_optimizer"].__name__, task["optimizee"].__name__))

    attack_model = task["attack_model"]()  # targeted model to attack
    if args.cuda:
        attack_model.cuda(args.gpu_num)
    ckpt_dict = torch.load(task["attack_model_ckpt"], map_location='cpu')
    attack_model.load_state_dict(ckpt_dict)
    attack_model.eval()
    attack_model.reset()  # not include parameters

    meta_model = task["optimizee"](optimizee.AttackModel(attack_model), task['batch_size'])  # meta optimizer
    if args.cuda:
        meta_model.cuda(args.gpu_num)
    train_loader, test_loader = meta_model.dataset_loader(args.data_dir, task['batch_size'], task['test_batch_size'])
    train_loader = iter(cycle(train_loader))

    if args.warm_start_ckpt != "None":
        meta_optimizer = task["nn_optimizer"](optimizee.MetaModel(meta_model), args, ckpt_path=args.warm_start_ckpt)
    else:
        meta_optimizer = task["nn_optimizer"](optimizee.MetaModel(meta_model), args)

    if args.cuda:
        meta_optimizer.cuda(args.gpu_num)
    optimizer = optim.Adam(meta_optimizer.parameters(), lr=task['lr'])

    min_test_loss = float("inf")

    for epoch in range(1, task["max_epoch"] + 1):
        decrease_in_loss = 0.0
        final_loss = 0.0
        meta_optimizer.train()
        for i in range(args.updates_per_epoch):
            # The `optimizee` for attack task
            model = task["optimizee"](optimizee.AttackModel(attack_model), task['batch_size'])
            if args.cuda:
                model.cuda(args.gpu_num)

            # In the attack task, each attacked image corresponds to a particular optmizee model
            data, target = next(train_loader)
            data, target = Variable(data.double()), Variable(target)
            if args.cuda:
                data, target = data.cuda(args.gpu_num), target.cuda(args.gpu_num)

            # Compute initial loss of the model
            f_x = model(data.double())
            initial_loss = model.loss(f_x, target)

            for k in range(task['optimizer_steps'] // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_state(
                    keep_states=k > 0, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda(args.gpu_num)
                for j in range(args.truncated_bptt_step):
                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the nn_optimizer
                    meta_model, *_ = meta_optimizer.meta_update(model, data, target)

                    # Compute a loss for a step the meta nn_optimizer
                    if not args.use_finite_diff:
                        # Use first-order method to train the zeroth-order optimizer
                        # (assume the gradient is available in training time)
                        f_x = meta_model(data)
                        loss = meta_model.loss(f_x, target)
                    else:
                        # Use zeroth-order method to train the zeroth-order optimizer
                        # Approximate the gradient
                        loss = optimizee.custom_loss(meta_model.weight, data, target, meta_model.nondiff_loss)

                    loss_sum += (k * args.truncated_bptt_step + j) * (loss - Variable(prev_loss))
                    prev_loss = loss.data

                    if hasattr(meta_optimizer, "reg_loss"):
                        loss_sum += meta_optimizer.reg_loss
                    if hasattr(meta_optimizer, "grad_reg_loss"):
                        loss_sum += meta_optimizer.grad_reg_loss

                # Update the parameters of the meta nn_optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for name, param in meta_optimizer.named_parameters():
                    if param.requires_grad:
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()

        # test
        meta_optimizer.eval()
        test_loss_sum = 0.0
        test_loss_ratio = 0.0
        num = 0
        for (test_data, test_target) in test_loader:
            test_data, test_target = Variable(test_data.double()), Variable(test_target)
            if args.cuda:
                test_data, test_target = test_data.cuda(args.gpu_num), test_target.cuda(args.gpu_num)
            model = task["optimizee"](optimizee.AttackModel(attack_model), task['test_batch_size'])
            if args.cuda:
                model.cuda(args.gpu_num)
            # Compute initial loss of the model
            f_x = model(test_data.double())
            test_initial_loss = model.loss(f_x, test_target)
            test_loss = 0.0

            meta_optimizer.reset_state(
                keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)

            for _ in range(task["test_optimizer_steps"]):
                _, test_loss, _ = meta_optimizer.meta_update(model, test_data, test_target)

            test_loss_sum += test_loss
            test_loss_ratio += test_loss / test_initial_loss
            num += 1

        msg = "Epoch: {}, final loss {}, average final/initial loss ratio: {}, test loss {}, test loss ratio {}".format(
            epoch,
            final_loss / args.updates_per_epoch,
            decrease_in_loss / args.updates_per_epoch,
            test_loss_sum / num, test_loss_ratio / num)
        print(msg)
        with open(os.path.join(args.output_dir, "train_log.txt"), 'a+') as f:
            f.write(msg + '\n')

        if epoch % args.epochs_per_ckpt == 0:
            meta_optimizer.save(epoch, args.output_dir)

        if test_loss_sum < min_test_loss:
            min_test_loss = test_loss_sum
            meta_optimizer.save(epoch, args.output_dir, best=True)


def optimizer_train_optimizee_attack(args):
    assert "Attack" in args.train_task
    task = train_task_list.tasks[args.train_task]

    attack_model = task["attack_model"]()
    if args.cuda:
        attack_model.cuda(args.gpu_num)
    ckpt_dict = torch.load(task["attack_model_ckpt"], map_location='cpu')
    attack_model.load_state_dict(ckpt_dict)
    attack_model.eval()
    attack_model.reset()  # not include parameters

    for test_idx in task['tests']['test_indexes']:
        _, test_loader = task["tests"]["optimizee"].dataset_loader(args.data_dir, task['batch_size'],
                                                                   task['tests']['test_batch_size'])
        test_loader = iter(test_loader)

        for _ in range(test_idx):  # attacked image
            data, target = next(test_loader)

        data, target = Variable(data.double()), Variable(target)
        if args.cuda:
            data, target = data.cuda(args.gpu_num), target.cuda(args.gpu_num)

        meta_model = task["tests"]["optimizee"](optimizee.AttackModel(attack_model), task['tests']['test_batch_size'])
        if args.cuda:
            meta_model.cuda(args.gpu_num)

        ckpt_path = os.path.join(args.output_dir, args.ckpt_path)

        # ZO-LSTM (leanred ZO optimizer)
        if "nn_opt" in task["tests"]:
            meta_optimizer = task["nn_optimizer"](optimizee.MetaModel(meta_model), args)
            if args.cuda:
                meta_optimizer.cuda(args.gpu_num)
            meta_optimizer.load(ckpt_path)
            meta_optimizer.eval()
            nn_opt_loss_array = []

        # ZO-SGD
        if "base_opt" in task["tests"]:
            base_optimizer = task["tests"]["base_opt"](None, args, task["tests"]["base_lr"])
            base_optimizer.eval()
            base_opt_loss_array = []

        # ZO-signSGD
        if "sign_opt" in task["tests"]:
            sign_optimizer = task["tests"]["sign_opt"](None, args, task["tests"]["sign_lr"])
            sign_optimizer.eval()
            sign_opt_loss_array = []

        # ZO-ADAM
        if "adam_opt" in task["tests"]:
            adam_optimizer = task["tests"]["adam_opt"](None, args, task["tests"]["adam_lr"], task["tests"]["adam_beta_1"],
                                                       task["tests"]["adam_beta_2"])
            adam_optimizer.eval()
            adam_opt_loss_array = []

        # ZO-LSTM-no-query (without QueryRNN)
        if "nn_opt_no_query" in task["tests"]:
            meta_model_2 = task["tests"]["optimizee"](optimizee.AttackModel(attack_model), task['tests']['test_batch_size'])
            if args.cuda:
                meta_model_2.cuda(args.gpu_num)

            nn_optimizer_no_query = task["tests"]["nn_opt_no_query"](optimizee.MetaModel(meta_model_2), args)
            if args.cuda:
                nn_optimizer_no_query.cuda(args.gpu_num)
            nn_optimizer_no_query.load(ckpt_path)
            nn_optimizer_no_query.eval()
            nn_opt_no_query_loss_array = []

        # ZO-LSTM-no-update (without UpdateRNN)
        if "nn_opt_no_update" in task["tests"]:
            meta_model_3 = task["tests"]["optimizee"](optimizee.AttackModel(attack_model), task['tests']['test_batch_size'])
            if args.cuda:
                meta_model_3.cuda(args.gpu_num)

            nn_optimizer_no_update = task["tests"]["nn_opt_no_update"](optimizee.MetaModel(meta_model_3), args)
            if args.cuda:
                nn_optimizer_no_update.cuda(args.gpu_num)
            nn_optimizer_no_update.load(ckpt_path)
            nn_optimizer_no_update.eval()
            nn_opt_no_update_loss_array = []

        # ZO-LSTM-guided (use Guided-ES to modify search distribution)
        if "nn_opt_guided" in task["tests"]:
            meta_model_4 = task["tests"]["optimizee"](optimizee.AttackModel(attack_model), task['tests']['test_batch_size'])
            if args.cuda:
                meta_model_4.cuda(args.gpu_num)

            nn_optimizer_guided = task["tests"]["nn_opt_guided"](optimizee.MetaModel(meta_model_4), args)
            if args.cuda:
                nn_optimizer_guided.cuda(args.gpu_num)
            nn_optimizer_guided.load(ckpt_path)
            nn_optimizer_guided.eval()
            nn_opt_guided_loss_array = []

        for num in range(1, task["tests"]["test_num"] + 1):
            model = task["tests"]["optimizee"](optimizee.AttackModel(attack_model), task['tests']['test_batch_size'])
            if args.cuda:
                model.cuda(args.gpu_num)

            if "nn_opt" in task["tests"]:
                meta_optimizer.reset_state(
                    keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)
                nn_opt_state = copy.deepcopy(model.state_dict())

            if "base_opt" in task["tests"]:
                base_opt_state = copy.deepcopy(model.state_dict())

            if "sign_opt" in task["tests"]:
                sign_opt_state = copy.deepcopy(model.state_dict())

            if "adam_opt" in task["tests"]:
                adam_optimizer.reset_state(
                    keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)
                adam_opt_state = copy.deepcopy(model.state_dict())

            if "nn_opt_no_query" in task["tests"]:
                nn_optimizer_no_query.reset_state(
                    keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)
                nn_opt_no_query_state = copy.deepcopy(model.state_dict())

            if "nn_opt_no_update" in task["tests"]:
                nn_optimizer_no_update.reset_state(
                    keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)
                nn_opt_no_update_state = copy.deepcopy(model.state_dict())

            if "nn_opt_guided" in task["tests"]:
                nn_optimizer_guided.reset_state(
                    keep_states=False, model=model, use_cuda=args.cuda, gpu_num=args.gpu_num)
                nn_opt_guided_state = copy.deepcopy(model.state_dict())

            for step in range(1, task["tests"]["n_steps"] + 1):
                msg = "iteration {}".format(step)

                # nn_opt
                if "nn_opt" in task["tests"]:
                    model.load_state_dict(nn_opt_state)
                    with torch.no_grad():
                        _, nn_opt_loss, nn_f_x = meta_optimizer.meta_update(model, data, target)
                    nn_opt_state = copy.deepcopy(model.state_dict())

                    msg += ", nn_opt_loss {:.6f}".format(nn_opt_loss.data.item())
                    nn_opt_loss_array.append(nn_opt_loss.data.item())

                # base_opt
                if "base_opt" in task["tests"]:
                    model.load_state_dict(base_opt_state)
                    with torch.no_grad():
                        _, base_opt_loss, base_f_x = base_optimizer.meta_update(model, data, target)
                    base_opt_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", base_opt_loss {:.6f}".format(base_opt_loss.data.item())
                    base_opt_loss_array.append(base_opt_loss.data.item())

                # sign_opt
                if "sign_opt" in task["tests"]:
                    model.load_state_dict(sign_opt_state)
                    with torch.no_grad():
                        _, sign_opt_loss, sign_f_x = sign_optimizer.meta_update(model, data, target)
                    sign_opt_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", sign_opt_loss {:.6f}".format(sign_opt_loss.data.item())
                    sign_opt_loss_array.append(sign_opt_loss.data.item())

                if "adam_opt" in task["tests"]:
                    model.load_state_dict(adam_opt_state)
                    with torch.no_grad():
                        _, adam_opt_loss, adam_f_x = adam_optimizer.meta_update(model, data, target)
                    adam_opt_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", adam_opt_loss {:.6f}".format(adam_opt_loss.data.item())
                    adam_opt_loss_array.append(adam_opt_loss.data.item())

                if "nn_opt_no_query" in task["tests"]:
                    model.load_state_dict(nn_opt_no_query_state)
                    with torch.no_grad():
                        _, nn_opt_no_query_loss, nn_no_query_f_x = nn_optimizer_no_query.meta_update(model, data, target,
                                                                                                     pred_query=False)
                    nn_opt_no_query_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", nn_opt_no_query_loss {:.6f}".format(nn_opt_no_query_loss.data.item())
                    nn_opt_no_query_loss_array.append(nn_opt_no_query_loss.data.item())

                if "nn_opt_no_update" in task["tests"]:
                    model.load_state_dict(nn_opt_no_update_state)
                    with torch.no_grad():
                        _, nn_opt_no_update_loss, nn_no_update_f_x = nn_optimizer_no_update.meta_update(model, data, target,
                                                                                                        pred_update=False,
                                                                                                        base_lr=
                                                                                                        task["tests"][
                                                                                                            "base_lr"])
                    nn_opt_no_update_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", nn_opt_no_update_loss {:.6f}".format(nn_opt_no_update_loss.data.item())
                    nn_opt_no_update_loss_array.append(nn_opt_no_update_loss.data.item())

                if "nn_opt_guided" in task["tests"]:
                    model.load_state_dict(nn_opt_guided_state)
                    with torch.no_grad():
                        _, nn_opt_guided_loss, nn_guided_f_x = nn_optimizer_guided.meta_update(model, data, target,
                                                                                               guided=True,
                                                                                               base_lr=
                                                                                               task["tests"][
                                                                                                   "base_lr"])
                    nn_opt_guided_state = copy.deepcopy(model.state_dict())
                    msg = msg + ", nn_opt_guided_loss {:.6f}".format(nn_opt_guided_loss.data.item())
                    nn_opt_guided_loss_array.append(nn_opt_guided_loss.data.item())
                print(msg)

            if args.save_loss:
                if "nn_opt" in task["tests"]:
                    np.save(
                        os.path.join(args.output_dir,
                                     "nn_opt_loss_array_{}_q_{}.npy".format(test_idx,
                                                                            args.grad_est_q)),
                        np.array(nn_opt_loss_array))
                if "base_opt" in task["tests"]:
                    np.save(
                        os.path.join(args.output_dir,
                                     "base_opt_loss_array_{}_q_{}.npy".format(test_idx,
                                                                              args.grad_est_q)),
                        np.array(base_opt_loss_array))
                if "sign_opt" in task["tests"]:
                    np.save(
                        os.path.join(args.output_dir,
                                     "sign_opt_loss_array_{}_q_{}.npy".format(test_idx,
                                                                              args.grad_est_q)),
                        np.array(sign_opt_loss_array))
                if "adam_opt" in task["tests"]:
                    np.save(
                        os.path.join(args.output_dir,
                                     "adam_opt_loss_array_{}_q_{}.npy".format(test_idx,
                                                                              args.grad_est_q)),
                        np.array(adam_opt_loss_array))
                if "nn_opt_no_query" in task["tests"]:
                    np.save(os.path.join(args.output_dir,
                                         "nn_opt_no_query_loss_array_{}_q_{}.npy".format(test_idx,
                                                                                         args.grad_est_q)),
                            np.array(nn_opt_no_query_loss_array))
                if "nn_opt_no_update" in task["tests"]:
                    np.save(os.path.join(args.output_dir,
                                         "nn_opt_no_update_loss_array_{}_q_{}.npy".format(test_idx,
                                                                                          args.grad_est_q)),
                            np.array(nn_opt_no_update_loss_array))
                if "nn_opt_guided" in task["tests"]:
                    np.save(os.path.join(args.output_dir,
                                         "nn_opt_guided_loss_array_{}_q_{}.npy".format(test_idx,
                                                                                       args.grad_est_q)),
                            np.array(nn_opt_guided_loss_array))
            print("Test num {}, test idx {}, done!".format(num, test_idx))

        if args.save_fig:
            assert args.save_loss
            fig = plt.figure(figsize=(8, 6))
            iteration = np.arange(1, task["tests"]["n_steps"] + 1)
            if "base_opt" in task["tests"]:
                base_opt_loss_array = np.load(os.path.join(args.output_dir,
                                                           "base_opt_loss_array_{}_q_{}.npy".format(
                                                               test_idx,
                                                               args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                base_opt_mean = np.mean(base_opt_loss_array, axis=0)
                base_opt_std = np.std(base_opt_loss_array, axis=0)
                plt.plot(iteration, base_opt_mean, 'c', label='ZO-SGD')
                plt.fill_between(iteration, base_opt_mean - base_opt_std, base_opt_mean + base_opt_std, color='c',
                                 alpha=0.2)

            if "sign_opt" in task["tests"]:
                sign_opt_loss_array = np.load(os.path.join(args.output_dir,
                                                           "sign_opt_loss_array_{}_q_{}.npy".format(
                                                               test_idx,
                                                               args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                sign_opt_mean = np.mean(sign_opt_loss_array, axis=0)
                sign_opt_std = np.std(sign_opt_loss_array, axis=0)
                plt.plot(iteration, sign_opt_mean, 'g', label='ZO-signSGD')
                plt.fill_between(iteration, sign_opt_mean - sign_opt_std, sign_opt_mean + sign_opt_std, color='g',
                                 alpha=0.2)

            if "adam_opt" in task["tests"]:
                adam_opt_loss_array = np.load(os.path.join(args.output_dir,
                                                           "adam_opt_loss_array_{}_q_{}.npy".format(
                                                               test_idx,
                                                               args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                adam_opt_mean = np.mean(adam_opt_loss_array, axis=0)
                adam_opt_std = np.std(adam_opt_loss_array, axis=0)
                plt.plot(iteration, adam_opt_mean, 'darkorange', label='ZO-ADAM')
                plt.fill_between(iteration, adam_opt_mean - adam_opt_std, adam_opt_mean + adam_opt_std, color='darkorange',
                                 alpha=0.2)

            if "nn_opt" in task["tests"]:
                nn_opt_loss_array = np.load(os.path.join(args.output_dir,
                                                         "nn_opt_loss_array_{}_q_{}.npy".format(test_idx,
                                                                                                args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                nn_opt_mean = np.mean(nn_opt_loss_array, axis=0)
                nn_opt_std = np.std(nn_opt_loss_array, axis=0)
                plt.plot(iteration, nn_opt_mean, 'b', label='ZO-LSTM')
                plt.fill_between(iteration, nn_opt_mean - nn_opt_std, nn_opt_mean + nn_opt_std, color='b', alpha=0.2)

            if "nn_opt_no_query" in task["tests"]:
                nn_opt_no_query_loss_array = np.load(os.path.join(args.output_dir,
                                                                  "nn_opt_no_query_loss_array_{}_q_{}.npy".format(
                                                                      test_idx,
                                                                      args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                nn_opt_no_query_mean = np.mean(nn_opt_no_query_loss_array, axis=0)
                nn_opt_no_query_std = np.std(nn_opt_no_query_loss_array, axis=0)
                plt.plot(iteration, nn_opt_no_query_mean, 'r', label='ZO-LSTM-no-query')
                plt.fill_between(iteration, nn_opt_no_query_mean - nn_opt_no_query_std,
                                 nn_opt_no_query_mean + nn_opt_no_query_std, color='r', alpha=0.2)

            if "nn_opt_no_update" in task["tests"]:
                nn_opt_no_update_loss_array = np.load(os.path.join(args.output_dir,
                                                                   "nn_opt_no_update_loss_array_{}_q_{}.npy".format(
                                                                       test_idx,
                                                                       args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                nn_opt_no_update_mean = np.mean(nn_opt_no_update_loss_array, axis=0)
                nn_opt_no_update_std = np.std(nn_opt_no_update_loss_array, axis=0)
                plt.plot(iteration, nn_opt_no_update_mean, 'm', label='ZO-LSTM-no-update')
                plt.fill_between(iteration, nn_opt_no_update_mean - nn_opt_no_update_std,
                                 nn_opt_no_update_mean + nn_opt_no_update_std, color='m', alpha=0.2)

            if "nn_opt_guided" in task["tests"]:
                nn_opt_guided_loss_array = np.load(os.path.join(args.output_dir,
                                                                "nn_opt_guided_loss_array_{}_q_{}.npy".format(
                                                                    test_idx,
                                                                    args.grad_est_q))).reshape(
                    (task["tests"]["test_num"], task["tests"]["n_steps"]))
                nn_opt_guided_mean = np.mean(nn_opt_guided_loss_array, axis=0)
                nn_opt_guided_std = np.std(nn_opt_guided_loss_array, axis=0)
                plt.plot(iteration, nn_opt_guided_mean, 'saddlebrown', label='ZO-LSTM-GuidedES')
                plt.fill_between(iteration, nn_opt_guided_mean - nn_opt_guided_std, nn_opt_guided_mean + nn_opt_guided_std,
                                 color='saddlebrown', alpha=0.2)
            plt.xlabel('iteration', fontsize=15)
            plt.ylabel('loss', fontsize=15)
            plt.legend(prop={'size': 15})
            fig.savefig(os.path.join(args.output_dir,
                                     args.fig_preffix + '_{}_q_{}.png'.format(test_idx, args.grad_est_q)))


def main(args):
    torch.set_default_dtype(torch.float64)
    if args.train == "optimizer_attack":  # train optimizer
        train_optimizer_attack(args)
    elif args.train == "optimizer_train_optimizee_attack":  # use the learned optimizer to train optimizee
        optimizer_train_optimizee_attack(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="output", help='output directory')
    parser.add_argument('--data_dir', type=str, default="data", help='data directory')
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--ckpt_path', type=str, help='checkpoint path')
    parser.add_argument('--warm_start_ckpt', type=str, default="None", help='checkpoint path for warm start')
    parser.add_argument('--train', type=str, default="optimizer_attack",
                        choices=["optimizer_attack", "optimizer_train_optimizee_attack"],
                        help='train optimizer or use the learned optimizer to train optimizee')
    parser.add_argument('--train_task', type=str, default="ZOL2L-Attack",
                        choices=train_task_list.tasks.keys(), help='MNIST only support `attack` yet')
    parser.add_argument('--grad_est_q', type=int, default=20, help='number of query directions for gradient estimation')
    parser.add_argument('--truncated_bptt_step', type=int, default=20, help='TBPTT steps')
    parser.add_argument('--updates_per_epoch', type=int, default=10, help='number of unrolled optimizations per epoch')
    parser.add_argument('--epochs_per_ckpt', type=int, default=1, help='number of epochs to save checkpoint')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--save_loss', action='store_true', help='save loss arrays')
    parser.add_argument('--save_fig', action='store_true', help='save loss curves')
    parser.add_argument('--fig_preffix', type=str, default='loss')
    parser.add_argument('--use_finite_diff', action='store_true',
                        help='use zeroth-order method to train the zeroth-order optimizer')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    main(args)
