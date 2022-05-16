#TODO 重复试验避免偶然性
import numpy as np
import paddle
import paddle.nn.functional as F

from .base import FinetuneBase 

import copy
class FinetuneDELTA(FinetuneBase):
    _confs = {'reg_weight': 0.01}
    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs = _confs):
        super(FinetuneDELTA, self).__init__(model, model_arch = model_arch, confs = confs)
        self.reg_weight = float(confs['reg_weight'])
        self.source_model = copy.deepcopy(model)
        self.source_model.eval()
   
    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        if self.model_arch == 'vit':
            fea_source = self.source_model.forward_features(x_data)
        else:
            _, fea_source = self.source_model(x_data)
            
        loss_reg = paddle.norm((features - fea_source.detach()) / features.shape[0])
        #print(epoch, batch_id, features.shape, loss_reg.item(), loss_ce.item())
        return {'loss_delta': self.reg_weight * loss_reg}

       
