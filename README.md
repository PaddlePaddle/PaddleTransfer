# PaddleTransfer
## Introduction
PaddleTransfer, a transfer learning tool of PaddlePaddle, provides a variety of algorithms to transfer the knowledge learned from source tasks to target task. It enables users to use the state-of-the-art transfer learning algorithms on main-stream model archtechtures.

PaddleTransfer provides various transfer learning algorithms, including **[MMD(JMLR'12)](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf?ref=https://githubhelp.com),[L2-SP(ICML'18)](http://proceedings.mlr.press/v80/li18a/li18a.pdf), [DELTA(ICLR'19)](https://openreview.net/pdf?id=rkgbwsAcYm), [RIFLE(ICML'20)](http://proceedings.mlr.press/v119/li20r/li20r.pdf), [Co-Tuning(NIPS'20)](https://proceedings.neurips.cc/paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf), [MARS-PGM(ICLR'21)](https://openreview.net/pdf?id=IFqrg1p5Bc)** and supports many main-stream model archtechtures, including **ResNet, MobileNet and ViT**. With several lines of codes, users can apply these algorithms on our predifined model or their own model easily.

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
### Install by pip
```
python -m pip install paddletransfer
```
### Dependencies
if you want to use our package in your own code, the following dependencies are required
* python >= 3.7
* numpy >= 1.21
* paddlepaddle >= 2.2 (with suitable CUDA and cuDNN version)

If you want to run our demo script, please make sure the following packages are installed correctly on your machine.
* visualdl
* yacs


## Usage Guideline
### Quick Start
Users can run our demo code for a quick start
```
python finetune.py --name [experiment_name] --train_dir [path_to_train_dir] --eval_dir [path_to_eval_dir] --model_arch [model_archtechture] --algo [finetune_algorithm] --gpu [device_for_experiment]
```
For model_arch argument, please choose one from "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "mobilenet_v2" and "vit". And for algo argument, please choose one from "base", "mmd", "rifle", "l2sp", "delta" and "cot". Mistyping may lead to unexpected behavior.

Please organize your dataset in the following format.
```
|_ root/
|  |_ class1
|  |  |_ image_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ class100
|     |_ ...
|     |_ image_9993.JPEG
```

If you want to finetune the ViT model, please make sure you have set the configuration file and pretrained parameters file correctly, and remember to add the corresponding argumengts(--cfg and --model_path) in your command. You can get the configuration file and pretrained model from [PaddleVit](https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification/ViT).

### Use PaddleTransfer in Your Own Code


#### Import dependencies
```
from paddletransfer.finetuning.img_cls import *
```
We strongly recommand you to use the model architechtures we provide in **backbones** by following codes. The parameters setting are the same as the implementations in [paddle.vision.models](https://github.com/PaddlePaddle/Paddle/tree/release/2.3/python/paddle/vision/models), remember to set **pretrained=True** for finetuning.
```
from backbones import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2, build_vit
```
You can only import the model archtechtures that you need.

If you want to use ViT model, please put the config.py file either from [our project](https://github.com/PaddlePaddle/PaddleTransfer) or [PaddleVit](https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification/ViT) under your working directory and set other files correctly as described in [Quick Start](#Quick-Start)

If you want to use your self-defined model, please make sure that the name of layers are consistent with the implementations in [paddle.vision.models](https://github.com/PaddlePaddle/Paddle/tree/release/2.3/python/paddle/vision/models) and your model has two outputs: The first one is the original output of the network and the second one is the features(the intermediate output before average pooling layer and fc layer).
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
algo = FinetuneBASE(model, model_arch)
```
### MMD
Use the following code for invoking MMD algorithm 
```
algo = FinetuneMMD(model, model_arch, confs=_confs)
```
The default hyperparameters for MMD algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.1, 'kernel_mul': 2.0, 'kernel_num': 5}
```

### L2-SP
Use the following code for invoking L2-SP algorithm 
```
algo = FinetuneL2SP(model, model_arch, confs=_confs)
```
The default hyperparameters for L2SP algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.01}
```

### DELTA
Use the following code for invoking DELTA algorithm 
```
algo = FinetuneDELTA(model, model_arch, confs=_confs)
```
The default hyperparameters for DELTA algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reg_weight': 0.01}
```

### RIFLE
Use the following code for invoking RIFLE algorithm 
```
algo = FinetuneRIFLE(model, model_arch, confs=_confs)
```
The default hyperparameters for RIFLE algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'reinit_epochs': [10,20,30]} # with total epochs = 40
```

### Co-Tuning

Use the following code for invoking Co-Tuning algorithm. The data_loader is used for relationship learning. 
```
algo = FinetuneCOT(model, model_arch, data_loader=data_loader, confs=_confs)
```
If you want to use Co-Tuning algorithm on ViT, use the following code. The vit_config is used for auxillary imagenet classifier construction. Please make sure that your vit_config object passed to the initializer has the same structure with the official implementation in [PaddleVit](https://github.com/BR-IDL/PaddleViT/blob/develop/image_classification/ViT/config.py), you can use the get_config() function they provide to generate one easily.
```
algo = FinetuneCOT(model, model_arch, data_loader=data_loader, vit_config=config, confs=_confs)
```
The default hyperparameters for Co-Tuning algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
 _confs = {'reg_weight': 2.3}
```
### MARS-PGM

Use the following code for invoking MARS-PGM algorithm. The data_loader is used for relationship learning. 
```
algo = FinetuneMARSPGM(model, model_arch, confs=_confs)
```
The default hyperparameters for MARS-PGM algorithms is as follows, if you want to modify them, please pass your own confs object to the initializer.
```
_confs = {'norm': 'mars', 'lambda_conv': float(20.), 'lambda_linear': float(20.), 'lambda_bn': float(20.)}
```

### Algorithm Performance
We have conducted some experiments on several dataset(**CUB_200_2011, indoorCVPR_09 and dtd**) using algorithms provided by PaddleTransfer. Most of experiments use the default hyper parameter setting in finetune.py, except Co-Tuning and MARS-PGM. For Co-Tuning, we use default hyper parameters on bird classification task, `lr = 0.002` and `wd = 0.0005` on scene classification task, `lr = 0.001` and `wd = 0.0005` on texture classification task. For MARS-PGM, we use `lambda_linear = 35` and `lambda_conv = lambda_bn = 20` on bird classification task, default hyper parameters on scene classification task, `lambda_linear = 15` and `lambda_conv = lambda_bn = 10` on texture classification task. The outcomes are as follows.



|          | Bird     | Scene    | Texture |
| -------- | -------- | -------- | ------- |
| BASE     | 80.25    | 77.34    | 62.07   |
| MMD      | 80.41    | 78.27    | 63.15   |
| L2SP     | 80.39    | 77.57    | 65.34   |
| DELTA    | 81.04    | 78.79    | 67.47   |
| RIFLE    | 80.74    | 78.08    | 63.15   |
| CoTuning | **81.07**    | 78.00    | **69.39**   |
| MARS-PGM | 81.01    | **78.87**    | 69.35   |
