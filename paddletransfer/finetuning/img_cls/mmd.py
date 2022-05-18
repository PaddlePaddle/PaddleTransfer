import numpy as np
import paddle
import paddle.nn.functional as F

from .base import FinetuneBase 

class MMD_loss(paddle.nn.Layer):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.shape[0])+int(target.shape[0])
        total = paddle.concat([source, target], axis=0)

        total0 = total.unsqueeze(0).expand([int(total.shape[0]), int(total.shape[0]), int(total.shape[1]), int(total.shape[2]), int(total.shape[3])])
        total1 = total.unsqueeze(1).expand([int(total.shape[0]), int(total.shape[0]), int(total.shape[1]), int(total.shape[2]), int(total.shape[3])])
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = paddle.sum(L2_distance) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [paddle.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.shape[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = paddle.mean(XX + YY - XY -YX)
        return loss

class FinetuneMMD(FinetuneBase):
    _confs = {'reg_weight': 0.1, 'kernel_mul': 2.0, 'kernel_num': 5}
    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs=_confs):
        super(FinetuneMMD, self).__init__(model, model_arch = model_arch, confs = confs)
        self.reg_weight = confs['reg_weight']
        self.kernel_mul = confs['kernel_mul']
        self.kernel_num = confs['kernel_num']
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((1, 1))
        self.mmd_loss = MMD_loss(self.kernel_mul, self.kernel_num)

    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        if self.model_arch != 'vit':
            features = self.avgpool(features)
        while len(features.shape)<4:
            features = features.unsqueeze(-1)
        loss_reg = paddle.to_tensor(0)
        bs = features.shape[0]
        drop = 0 if bs % 2 == 0 else 1
        loss_reg = self.mmd_loss(features[:bs//2], features[bs//2:bs-drop])
        return {'loss_mmd': self.reg_weight * loss_reg}
