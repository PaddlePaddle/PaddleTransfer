import numpy as np
import paddle
import paddle.nn.functional as F

class FinetuneBase: 
    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs=None):
        self.model = model
        self.model_arch = model_arch

    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        return {}

    def params(self):
        return self.model.parameters()