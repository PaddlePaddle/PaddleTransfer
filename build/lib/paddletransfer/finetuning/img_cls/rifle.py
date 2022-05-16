#TODO 首次实验进行中ing...
import math
import numpy as np
import paddle
import paddle.nn.functional as F

from .base import FinetuneBase 

class FinetuneRIFLE(FinetuneBase):
    _confs = {'reinit_epochs': [10,20,30]}
    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs = _confs):
        super(FinetuneRIFLE, self).__init__(model, model_arch = model_arch, confs = confs)
        self.reinit_epochs = confs['reinit_epochs']

    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        if epoch in self.reinit_epochs and epoch > -1 and batch_id == 0:
            if self.model_arch == "vit":
                param_shape = self.model.classifier.weight.shape
                stdv = 1. / math.sqrt(param_shape[1])
                self.model.classifier.weight.data = stdv * paddle.uniform(shape=self.model.classifier.weight.shape)
            else:
                param_shape = self.model.fc.weight.shape
                stdv = 1. / math.sqrt(param_shape[1])
                self.model.fc.weight.data = stdv * paddle.uniform(shape=self.model.fc.weight.shape)
        return {}

