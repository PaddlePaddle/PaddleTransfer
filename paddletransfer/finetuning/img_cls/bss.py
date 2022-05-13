import numpy as np
import paddle
import paddle.nn.functional as F

from .base import FinetuneBase 

import copy
class FinetuneBSS(FinetuneBase):
    _confs = {'reg_weight': 0.001, 'k': 1}

    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs=_confs):
        super(FinetuneBSS, self).__init__(model, model_arch)
        self.reg_weight = float(confs['reg_weight'])
        self.k = float(confs['k'])
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((1, 1)) 


    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return 


    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        loss_reg = paddle.to_tensor(0)
        features = self.avgpool(features)
        features = paddle.flatten(features, 1)
        print(features.t().shape)
        u, s, vt = paddle.linalg.svd(features.t())
        print(u.shape, s.shape, vt.shape)
        print('s', s)
        num = s.size(0)
        for i in range(self.k):
            loss_reg += paddle.pow(s[num-1-i], 2)
        return {'loss_bss': self.reg_weight * loss_reg}

