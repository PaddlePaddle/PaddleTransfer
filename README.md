# PaddleTransfer
## Introduction
PaddleTransfer, a transfer learning tool of PaddlePaddle, provides a variety of algorithms to transfer the knowledge learned from source tasks to target task. It enables users to use the state-of-the-art transfer learning algorithms on main-stream model archtechtures.

PaddleTransfer provides various transfer learning algorithms, including **MMD(JMLR'12),L2-SP(ICML'18), DELTA(ICLR'19), RIFLE(ICML'20), Co-Tuning(NIPS'20)** and supports many main-stream model archtechtures, including **ResNet, MobileNet and ViT**. With several lines of codes, users can apply these algorithms on our predifined model or their own model easily.

## Contents
* [Key Highlights](#Key-Highlights)
* [Installation](#Installation)
* [Usage Guideline](#Usage-Guideline)
* [Provided Algorithms](#Provided-Algorithms)

## Key Highlights
### High Accuracy
PaddleTransfer provides various transfer learning algorithms, users can conveniently apply them on their models, and select the most appropriate one with high accuracy for further usage.

### Easy to Use
PaddleTransfer provides unified API to invoke different algorithms, users can easily apply them on their models with several lines of codes.

### Fully Support
PaddleTransfer supports most commonly used models including Resnet, MobileNet and ViT, and will iterates rapidly to support more model archtectures.

## Installation

## Usage Guideline
### Quick Start
Users can run our demo code for a quick start
```
python finetune.py --name [experiment_name] --train_dir [path_to_train_dir] --eval_dir [path_to_eval_dir] --model_arch [model_archtechture] --algo [finetune_algorithm] --gpu [device_for_experiment]
```
For model_arch argument, please choose one from "resnet", "mobilenet_v2" and "vit", mistyping may lead to unexpected behavior.

If you want to finetune the ViT model on your dataset, please make sure you have set the configuration file and pretrained parameters file correctly, and remember to add the corresponding argumengts(--cfg and --model_path) in your command. You can get the configuration file and pretrained model from [PaddleVit](https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification/ViT).

### Use PaddleTransfer in Your Own Code
#### Import dependencies
```
from paddletransfer.finetuning.img_cls import *
```
#### Initialize the algorithm object
```
algo = FinetuneDELTA(model, model_arch)
```
* To customize the hyperparameter for finetune algorithm, please add the following arguments to the initializer, for details about hyperparameter setting, please refer to [provided algorithms](#Provided-Algorithms)
```
confs = confs # a directory of the hyperparameter setting
```

#### Get the parameters list which need to update and pass them to optimizer
```
params = algo.params()
...
optimizer = paddle.optimizer.Optimizer(...,parameters = params,...)
```
In most cases, it is equal to model.parameters() and you may not want to invoke this method. But remember to do this if you are using Co-tuning algorithm since it has extra parameters to update

#### Get the regularization loss and merge it to the classification loss
```
loss_cls = paddle.nn.CrossEntropyLoss(y_data,logits)
loss_all = {'loss_cls': loss_cls}
loss_reg = algo.loss(x_data, y_data, logits, features, epoch, batch_id)
loss_all.update(loss_reg)
loss = sum(loss_all.values())
...
loss.backward()
```
## Provided Algorithms
So far we have provided 5 algorithms for finetune, which are **MMD, L2-SP, DELTA, RIFLE and Co-Tuning**. If you do not want to use any finetune algorithms, just use the following code for vanilla finetune, the corresponding code for invoking different algorithms are in the respective sections.
```
algo = FinetuneBASE(model,model_arch)
```
### MMD
Use the following code for invoking MMD algorithm 
```
algo = FinetuneMMD(model,model_arch,confs=_confs)
```
The default hyperparameters for MMD algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.1, 'kernel_mul': 2.0, 'kernel_num': 5}
```

### L2-SP
Use the following code for invoking L2-SP algorithm 
```
algo = FinetuneL2SP(model,model_arch,confs=confs)
```
The default hyperparameters for L2SP algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.01}
```

### DELTA
Use the following code for invoking DELTA algorithm 
```
algo = FinetuneDELTA(model,model_arch,confs=confs)
```
The default hyperparameters for DELTA algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.01}
```

### RIFLE
Use the following code for invoking RIFLE algorithm 
```
algo = FinetuneRIFLE(model,model_arch,confs=confs)
```
The default hyperparameters for RIFLE algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reinit_epochs': [10,20,30]} # with total epochs = 40
```

### Co-Tuning

Use the following code for invoking Co-Tuning algorithm. The data_loader is used for relationship learning. 
```
algo = FinetuneCOT(model, model_arch, data_loader=data_loader, confs=confs)
```
If you want to use Co-Tuning algorithm on ViT, use the following code. The vit_config is used for auxillary imagenet classifier construction.
```
algo = FinetuneCOT(model, model_arch, data_loader=data_loader, vit_config=config, confs=confs)
```
The default hyperparameters for Co-Tuning algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
 _confs = {'reg_weight': 2.3}
```

### Algorithm Performance
We have conducted some experiments on several dataset using algorithms provided by PaddleTransfer. Most of experiments use the default hyper parameter setting in finetune.py, except Co-Tuning. For Co-Tuning, We use default hyper parameters on bird classification task, lr = 0.002 and wd = 0.0005 on scene classification task, lr = 0.001 and wd = 0.0005 on texture classification task. The outcomes are as follows.



|          | Bird     | Scene    | Texture |
| -------- | -------- | -------- | ------- |
| BASE     | 80.25    | 77.34    | 62.07   |
| MMD      | 80.41    | 78.27    | 63.15   |
| L2SP     | 80.39    | 77.57    | 65.34   |
| DELTA    | 81.04    | **78.79**    | 67.47   |
| RIFLE    | 80.74    | 78.08    | 63.15   |
| CoTuning | **81.07**    | 78.00    | **69.39**   |

 ![img](imgs/result.jpg)