#TODO 找出loss_reg不变的原因
import numpy as np
import paddle
import paddle.nn.functional as F

from .base import FinetuneBase 

import copy
class FinetuneL2SP(FinetuneBase):
    _confs = {'reg_weight': 0.01}

    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs=_confs):
        super(FinetuneL2SP, self).__init__(model, model_arch = model_arch, confs = confs)
        self.reg_weight = float(confs['reg_weight'])
        self.pre_trained_weights = {}
        for name, param in self.model.named_parameters():
            self.pre_trained_weights[name] = copy.deepcopy(param.detach())
   
    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        loss_reg = paddle.to_tensor(0)
        for name, param in self.model.named_parameters():
            if self.model_arch == 'vit':
                if name.startswith('classifier'):
                    loss_reg += paddle.norm(param)
                else:
                    loss_reg += paddle.norm(param - self.pre_trained_weights[name])
            else:
                if name.startswith('fc'):
                    loss_reg += paddle.norm(param)
                else:
                    loss_reg += paddle.norm(param - self.pre_trained_weights[name])
        return {'loss_l2sp': self.reg_weight * loss_reg}
