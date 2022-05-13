import time
import logging
import os
import sys
import math
import argparse
import random
import numpy as np
from visualdl import LogWriter

import paddle
import paddle.nn.functional as F
from paddle.vision import transforms
from paddle.vision.datasets import DatasetFolder

from config import get_config
from backbones import build_vit
from paddletransfer.finetuning.img_cls import *



def get_args():
    parser = argparse.ArgumentParser(description = 'PaddlePaddle Deep Transfer Learning Toolkit, Image Classification Fine-tuning Example')
    
    parser.add_argument('--name', type = str, default = 'flower102')
    parser.add_argument('--train_dir', default = '../CoTuning/data/finetune/flower102/train')
    parser.add_argument('--eval_dir', default = '../CoTuning/data/finetune/flower102/test')
    parser.add_argument('--log_dir', default = './visual_log')
    parser.add_argument('--save', type = str, default = './output')
    

    parser.add_argument('--image_size', type = int, default = 384)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--epochs', type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--model_path',default = './models/vit_base_patch16_384.pdparams')
    parser.add_argument('--cfg', type = str, default = "./configs/vit_base_patch16_384.yaml")

    parser.add_argument('--algo', type=str, default = 'base')
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--print_frequency', type=int, default=100)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--save_frequency',default = 10)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args


def custom_config(args):
    assert os.path.exists(args.cfg), "Please set the configuration file first."
    config = get_config(args.cfg)
    config.defrost()
    config.DATA.DATA_PATH = args.train_dir
    config.DATA.DATASET = args.name
    config.DATA.IMAGE_SIZE = args.image_size
    config.DATA.BATCH_SIZE = args.batch_size
    config.DATA.BATCH_SIZE_EVAL = args.batch_size_eval
    config.TRAIN.NUM_EPOCHS = args.epochs   # TODO 500 steps
    config.TRAIN.WEIGHT_DECAY = args.wd
    config.TRAIN.BASE_LR = args.lr
    config.TRAIN.OPTIMIZER = 'Momentum'
    config.REPORT_FREQ = args.print_frequency  # freq to logging info
    config.VALIDATE_FREQ = args.eval_frequency  # freq to do validation
    config.SAVE = args.save
    config.SEED = args.seed
    if args.eval_dir:
        config.EVAL = True
    
    config.freeze()
    return config


def load_model(args, config):
    paddle_model = build_vit(config)
    assert os.path.exists(args.model_path), "Please download the pretrained model first."
    model_state_dict = paddle.load(args.model_path)
    paddle_model.set_state_dict(model_state_dict)
    return paddle_model

def get_dataloader_train(args,config):
    train_path = args.train_dir
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=(config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
                                     interpolation='bicubic'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    train_set = DatasetFolder(train_path, transform=transform_train)
    train_loader = paddle.io.DataLoader(train_set, shuffle=True, batch_size=config.DATA.BATCH_SIZE)
    num_classes = len(train_set.classes)

    return train_loader, num_classes

def get_dataloader_val(args,config):
    val_path = args.eval_dir
    scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
    transform_val = transforms.Compose([
        transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
        transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    val_set = DatasetFolder(val_path, transform=transform_val)
    val_loader = paddle.io.DataLoader(val_set, shuffle=False, batch_size=config.DATA.BATCH_SIZE_EVAL)

    return val_loader


def determine_algo(model, args, config, train_loader):
    if args.algo == 'base':
        algo = FinetuneBase(model, args.model_arch)
    elif args.algo == 'l2sp':
        algo = FinetuneL2SP(model, args.model_arch)
    elif args.algo == 'delta':
        algo = FinetuneDELTA(model, args.model_arch)
    elif args.algo == 'rifle':
        algo = FinetuneRIFLE(model, args.model_arch, confs = {'reinit_epochs': [int(args.epochs*0.25),int(args.epochs*0.5),int(args.epochs*0.75)]})
    elif args.algo == 'mmd':
        algo = FinetuneMMD(model, args.model_arch)
    elif args.algo == 'cot':
        algo = FinetuneCOT(model, args.model_arch, data_loader = train_loader, vit_config = config)
    return algo


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


def train(dataloader,
          model,
          criterion,
          algo,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          amp=False,
          logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        amp: bool, if True, use mix precision training, default: False
        logger: logger for logging, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average top1 accuracy on current process/gpu
        train_time: float, training time
    """
    model.train()
    time_st = time.time()
    accuracies = []
    losses = []

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = paddle.unsqueeze(data[1], 1)
        features = model.forward_features(image)
        logits = model.classifier(features)

        loss_ce = criterion(logits, label)
        loss_all = {'loss_ce': loss_ce}
        loss_reg = algo.loss(image, label, logits, features, epoch, batch_id)
        loss_all.update(loss_reg)
        loss = sum(loss_all.values())
        losses.append(loss.numpy())

        if logger and batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Loss is: {loss.numpy()}, " +
                f"loss_all: {loss_all}")

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        acc = paddle.metric.accuracy(logits, label)
        accuracies.append(acc.numpy())
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

    train_time = time.time() - time_st
    return avg_loss, avg_acc, train_time


def validate(dataloader, model, criterion, total_batch, debug_steps=100, logger=None):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        val_acc5_meter.avg: float, average top5 accuracy on current process/gpu
        val_time: float, valitaion time
    """
    model.eval()
    losses = []
    top1_accuracies = []
    top5_accuracies = []
    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            label = paddle.unsqueeze(data[1], 1)
            logits = model(image)

            loss = criterion(logits, label)
            acc1 = paddle.metric.accuracy(logits, label)
            acc5 = paddle.metric.accuracy(logits, label, k=5)
            top1_accuracies.append(acc1.numpy())
            top5_accuracies.append(acc5.numpy())
            losses.append(loss.numpy())

            avg_acc1, avg_acc5, avg_loss = np.mean(top1_accuracies), np.mean(top5_accuracies), np.mean(losses)

            if logger and batch_id % debug_steps ==0 and batch_id != 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {avg_loss}, " +
                    f"Avg Acc@1: {avg_acc1}, " +
                    f"Avg Acc@5: {avg_acc5}")

    val_time = time.time() - time_st
    return avg_loss, avg_acc1, avg_acc5, val_time


def finetune_vit(args):
    # STEP 0: Preparation
    
    config = custom_config(args)
    last_epoch = -1
    paddle.device.set_device(f'gpu:{args.gpu}')
    seed = config.SEED
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)
    logger = get_logger(filename=os.path.join(config.SAVE, f'{args.name}_{args.algo}.txt'))
    logdir = os.path.join(args.log_dir,args.algo)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = LogWriter(logdir = logdir)
    logger.info(f'\n{args}')
    

    # STEP 1: Create train and val dataloader
    dataloader_train, num_classes = get_dataloader_train(args,config)
    if args.eval_dir:
        dataloader_val = get_dataloader_val(args,config)
    print(f"------------num classes:{num_classes}")

    # STEP 2: Load model
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.freeze()
    logger.info(f'\n{config}')
    model = load_model(args, config)
    logger.info('finish load the pretrained model')

    # STEP 3: determine algorithm for finetune
    algo = determine_algo(model, args, config, dataloader_train)

    # STEP 4: Define optimizer and lr_scheduler
    criterion = paddle.nn.CrossEntropyLoss()
    # warmup + cosine lr scheduler
    params = algo.params()
    cosine_lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                        learning_rate=config.TRAIN.BASE_LR,
                        T_max=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                        eta_min=config.TRAIN.END_LR,
                        last_epoch=-1)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=cosine_lr_scheduler,  # use cosine lr sched after warmup
                warmup_steps=config.TRAIN.WARMUP_EPOCHS,    # only support position integet
                start_lr=config.TRAIN.WARMUP_START_LR,
                end_lr=config.TRAIN.BASE_LR,
                last_epoch=config.TRAIN.LAST_EPOCH)
    # set gradient clip
    clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    optimizer = paddle.optimizer.Momentum(
        parameters=params,
        learning_rate=lr_scheduler, # set to scheduler
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        momentum=0.9,
        grad_clip=clip)

    # STEP 5: Run training
    logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_acc, train_time = train(dataloader=dataloader_train,
                                                  model=model,
                                                  criterion=criterion,
                                                  algo = algo,
                                                  optimizer=optimizer,
                                                  epoch=epoch,
                                                  total_epochs=config.TRAIN.NUM_EPOCHS,
                                                  total_batch=len(dataloader_train),
                                                  debug_steps=config.REPORT_FREQ,
                                                  accum_iter=config.TRAIN.ACCUM_ITER,
                                                  amp=config.AMP,
                                                  logger=logger)
        lr_scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")
        writer.add_scalar(tag="train_acc", step=epoch, value=train_acc)
        writer.add_scalar(tag="train_loss", step=epoch, value=train_loss)
        
        # validation
        if args.eval_dir:
            if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                logger.info(f'----- Validation after Epoch: {epoch}')
                val_loss, val_acc1, val_acc5, val_time = validate(
                    dataloader=dataloader_val,
                    model=model,
                    criterion=criterion,
                    total_batch=len(dataloader_val),
                    debug_steps=config.REPORT_FREQ,
                    logger=logger)
                logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                            f"Validation Loss: {val_loss:.4f}, " +
                            f"Validation Acc@1: {val_acc1:.4f}, " +
                            f"Validation Acc@5: {val_acc5:.4f}, " +
                            f"time: {val_time:.2f}")
                writer.add_scalar(tag="val_acc", step=epoch, value=val_acc)
                writer.add_scalar(tag="val_loss", step=epoch, value=val_loss)

        if epoch % args.save_frequency == 0 or epoch == args.epochs:
            model_path = os.path.join(args.save, f"Epoch{epoch}.pdparams")
            state_dict = dict()
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['epoch'] = epoch
            if lr_scheduler is not None:
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            paddle.save(state_dict, model_path)
            logger.info(f"----- Save model: {model_path}")


if __name__ == '__main__':
    print(paddle.__version__)
    args = get_args()
    finetune_vit()
