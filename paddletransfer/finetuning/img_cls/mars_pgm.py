import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from .base import FinetuneBase 

import copy
class FinetuneMARSPGM(FinetuneBase):
    '''
    MARS-PGM for CNN
    '''
    _confs={'norm': 'mars', 'lambda_conv': float(20.), 'lambda_linear': float(20.), 'lambda_bn': float(20.)}

    def __init__(self, model, model_arch, data_loader = None, vit_config = None, confs=_confs):
        super(FinetuneMARSPGM, self).__init__(model, model_arch = model_arch, confs=confs)
        self.norm = confs['norm']
        pre_trained_model = copy.deepcopy(model)
        for name, param in pre_trained_model.named_parameters():
            param = param.detach()
        for sublayer, pre_trained_sublayer in zip(self.model.sublayers(), pre_trained_model.sublayers()):
            if isinstance(sublayer, nn.Linear) and confs['lambda_linear'] != float('inf'):
                self._add_constraint(sublayer, confs['lambda_linear'])
            elif isinstance(sublayer, nn.Conv2D) and confs['lambda_conv'] != float('inf'):
                self._add_constraint(sublayer, confs['lambda_conv'], pre_trained_sublayer=pre_trained_sublayer)
            elif isinstance(sublayer, nn.BatchNorm2D) and confs['lambda_bn'] != float('inf'):
                self._add_bn_constraint(sublayer, confs['lambda_bn'], pre_trained_sublayer=pre_trained_sublayer)
   
    def _add_constraint(self, sublayer, max_k, pre_trained_sublayer=None, name="weight"):
        MARSNormConstraint.apply(sublayer, name, max_k, pre_trained_sublayer)
        return sublayer

    def _add_bn_constraint(self, sublayer, max_k, pre_trained_sublayer=None, name="weight"):
        MARSNormBnConstraint.apply(sublayer, name, max_k, pre_trained_sublayer)
        return sublayer


class MARSNormConstraint(object):
    def __init__(self, name, max_k, pre_trained_layer) -> None:
        self.name = name
        self.max_k = max_k
        self.pre_trained_weight = 0 if pre_trained_layer is None else getattr(pre_trained_layer, name)

    def _compute_weight(self, layer):
        weight = getattr(layer, self.name + '_orig')
        weight_zero = weight - self.pre_trained_weight
        axes = 0
        if len(weight.shape) == 4:
            axes = [1, 2, 3]
        norms = paddle.sum(paddle.abs(weight_zero), axis=axes, keepdim=True)
        weight_zero *= (1.0 / paddle.maximum(norms / self.max_k, paddle.to_tensor(1.0)))
        weight.set_value(weight_zero + self.pre_trained_weight)
        return weight

    @staticmethod
    def apply(layer, name, max_k, pre_trained_layer):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, MARSNormConstraint) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two MARSNormConstraint hooks on "
                    "the same parameter {}".format(name)
                )

        fn = MARSNormConstraint(name, max_k, pre_trained_layer)
        weight = layer._parameters[name]
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + '_orig', weight)
        setattr(layer, fn.name, weight)
        layer.register_forward_pre_hook(fn)
        return fn
    
    def __call__(self, layer, input):
        setattr(layer, self.name, self._compute_weight(layer))

class MARSNormBnConstraint(object):
    def __init__(self, name, max_k, pre_trained_layer) -> None:
        self.name = name
        self.max_k = max_k
        self.pre_trained_diag = 0 if pre_trained_layer is None else getattr(pre_trained_layer, self.name) / paddle.sqrt(getattr(pre_trained_layer, '_variance') + 1e-6)

    def _compute_weight(self, layer):
        weight = getattr(layer, self.name)
        variance = getattr(layer, '_variance')
        diag = weight / paddle.sqrt(variance + 1e-6)
        diag -= self.pre_trained_diag
        diag *= (1.0 / paddle.maximum(paddle.abs(weight) / self.max_k, paddle.to_tensor(1.0)))
        diag += self.pre_trained_diag
        weight.set_value(diag * paddle.sqrt(variance + 1e-6))
        return weight

    @staticmethod
    def apply(layer, name, max_k, pre_trained_layer):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, MARSNormBnConstraint) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two MARSNormBnConstraint hooks on "
                    "the same parameter {}".format(name)
                )

        fn = MARSNormBnConstraint(name, max_k, pre_trained_layer)
        weight = layer._parameters[name]
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + '_orig', weight)
        setattr(layer, fn.name, weight)
        layer.register_forward_pre_hook(fn)
        return fn
    
    def __call__(self, layer, input):
        setattr(layer, self.name, self._compute_weight(layer))
