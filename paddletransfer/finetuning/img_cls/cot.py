#TODO 接上VIT
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from backbones import mobilenet_v2,resnet34,resnet50,resnet101,build_vit
from tqdm import tqdm
import copy

from .base import FinetuneBase


class Head(nn.Layer):

    def __init__(self, model_arch, config):
        super(Head, self).__init__(model_arch, config)
        self.model_arch = model_arch

        if self.model_arch == 'vit':
            imagenet_config = copy.deepcopy(config)
            imagenet_config.defrost()
            imagenet_config.DATA.NUM_CLASSES = 1000
            imagenet_config.freeze()
            backbone = build_vit(imagenet_config)
        else:
            backbone = eval(model_arch)(pretrained=True)

        if model_arch == 'vit':
            self.fc = backbone.classifier
        elif model_arch == "mobilenet_v2":
            self.avgpool = backbone.pool2d_avg
            self.dropout = backbone.dropout
            self.fc = backbone.fc
        else:
            self.avgpool = backbone.avgpool
            self.fc = backbone.fc

    def forward(self, x):
        if self.model_arch != 'vit':
            x = self.avgpool(x)
            x = paddle.flatten(x, 1)
            if self.model_arch == "mobilenet_v2":
                x = self.dropout(x)
        x = self.fc(x)
        return x


class FinetuneCOT(FinetuneBase):
    _confs = {'reg_weight': 2.3}

    def __init__(self, model, model_arch, data_loader=None, vit_config=None, confs=_confs):
        super(FinetuneCOT, self).__init__(model, model_arch=model_arch, confs=confs)
        self.reg_weight = float(confs["reg_weight"])
        self.model_arch = model_arch
        self.head = Head(model_arch, vit_config)
        self.relationship = self.relationship_learning(data_loader)

    @paddle.no_grad()
    def relationship_learning(self, data_loader):
        print('computing relationship')

        train_labels_list = []
        imagenet_labels_list = []

        for train_inputs, train_labels in tqdm(data_loader):
            self.model.eval()
            self.head.eval()

            train_inputs = paddle.to_tensor(train_inputs)

            if self.model_arch != 'vit':
                _, features = self.model(train_inputs)
            else:
                features = self.model.forward_features(train_inputs)

            imagenet_labels = self.head(features)
            imagenet_labels = imagenet_labels.detach().numpy()

            train_labels_list.append(train_labels.numpy())
            imagenet_labels_list.append(imagenet_labels)

        labels = np.concatenate(train_labels_list, 0)
        logits = np.concatenate(imagenet_labels_list, 0)

        def softmax_np(x):
            max_el = np.max(x, axis=1, keepdims=True)
            x = x - max_el
            x = np.exp(x)
            s = np.sum(x, axis=1, keepdims=True)
            return x / s

        probabilities = softmax_np(logits * 0.8840456604957581)
        N_t = np.max(labels) + 1
        conditional = []
        for i in range(N_t):
            this_class = probabilities[labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)

        return np.concatenate(conditional)

    def loss(self, x_data, y_data, logits, features, epoch, batch_id):
        imagenet_targets = paddle.to_tensor([self.relationship[train_label] for train_label in y_data])
        
        self.head.train()
        imagenet_outputs = self.head(features)

        imagenet_loss = - imagenet_targets * F.log_softmax(imagenet_outputs)
        imagenet_loss = paddle.mean(paddle.sum(imagenet_loss, axis=-1))
        return {"imagenet_loss": self.reg_weight * imagenet_loss}

    def params(self):
        if self.model_arch == 'vit':
            parameters = [{'params': self.model.position_embedding},
                          {'params': self.model.cls_token},
                          {'params': self.model.patch_embedding.parameters()},
                          {'params': self.model.encoder.parameters()},
                          {'params': self.model.classifier.parameters(), 'learning_rate':10.0},
                          {'params': self.head.parameters()}]
        elif self.model_arch == 'mobilenet_v2':
            parameters = [{'params': self.model.features.parameters()},
                          {'params': self.model.fc.parameters(),'learning_rate':10.0},
                          {'params': self.head.parameters()}]
        else:
            parameters = [{'params': self.model.conv1.parameters()},
                          {'params': self.model.bn1.parameters()},
                          {'params': self.model.layer1.parameters()},
                          {'params': self.model.layer2.parameters()},
                          {'params': self.model.layer3.parameters()},
                          {'params': self.model.layer4.parameters()},
                          {'params': self.model.fc.parameters(), 'learning_rate':10.0},
                          {'params': self.head.parameters()}]
        return parameters
