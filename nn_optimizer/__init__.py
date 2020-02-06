import torch
from torch import nn

import os


class NNOptimizer(nn.Module):
    def __init__(self, model, args):
        super(NNOptimizer, self).__init__()
        self.meta_model = model

    def reset_state(self):
        raise NotImplementedError

    def meta_update(self, *input):
        raise NotImplementedError

    def GradientEstimate(self, model, data, target, direction, mu=0.000001):
        model_params = model.get_params()
        updated_model_params = model_params + mu * direction

        f_x1 = model(data)
        loss1 = model.loss(f_x1, target)

        model.set_params(updated_model_params)
        f_x2 = model(data)
        loss2 = model.loss(f_x2, target)

        grads = (loss2 - loss1) / mu
        if not self.training:
            grads.detach_()
        # grads.detach_()
        model.set_params(model_params)
        return grads

    def save(self, epoch, outdir, best=False):
        if not best:
            filepath = os.path.join(outdir, "ckpt_{}".format(epoch))
            print("Saving model state at epoch {} to {}".format(epoch, filepath))
        else:
            filepath = os.path.join(outdir, "ckpt_best")
            print("Saving model state at epoch {} to {}".format(epoch, filepath))
        torch.save({'epoch': epoch,
                    'state_dict': self.state_dict()}, filepath)

    def load(self, ckpt_path):
        assert os.path.isfile(ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt_dict['state_dict'])
        # print(ckpt_dict['state_dict'])
        epoch = ckpt_dict['epoch']
        print("Loaded checkpoint '{}' from epoch {}".format(ckpt_path, epoch))


from . import zoopt
from . import basezoopt
