import time
import logging
import os
import sys
import argparse
import random
from visualdl import LogWriter
import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.vision import transforms
from paddle.vision.datasets import DatasetFolder

from paddletransfer.finetuning.img_cls import *
from backbones import mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152


def get_args():
    parser = argparse.ArgumentParser(description='PaddlePaddle Deep Transfer Learning Toolkit, Image Classification Fine-tuning Example')
    parser.add_argument('--name', type = str, default = 'flower102')
    parser.add_argument('--train_dir', default='../CoTuning/data/finetune/flower102/train')
    parser.add_argument('--eval_dir', default='../CoTuning/data/finetune/flower102/test')
    parser.add_argument('--log_dir', default = './visual_log')
    parser.add_argument('--save', type = str, default = './output')

    parser.add_argument('--model_arch', default='resnet50')
    parser.add_argument('--image_size', type = int, default = 224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument('--algo', default='base')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_frequency', type=int, default=100)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--save_frequency', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args


def get_dataloader_train(args):
    train_path = args.train_dir
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = DatasetFolder(train_path, transform=transform_train)
    train_loader = paddle.io.DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    num_classes = len(train_set.classes)

    return train_loader, num_classes


def get_dataloader_val(args):
    val_path = args.eval_dir
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_set = DatasetFolder(val_path, transform=transform_val)
    val_loader = paddle.io.DataLoader(val_set, shuffle=False, batch_size=args.batch_size_eval)

    return val_loader


def determine_algo(model, args, train_loader):
    if args.algo == 'base':
        algo = FinetuneBase(model, model_arch = args.model_arch)
    elif args.algo == 'l2sp':
        algo = FinetuneL2SP(model, model_arch = args.model_arch)
    elif args.algo == 'delta':
        algo = FinetuneDELTA(model,model_arch = args.model_arch)
    elif args.algo == 'rifle':
        algo = FinetuneRIFLE(model, model_arch = args.model_arch)
    elif args.algo == 'mmd':
        algo = FinetuneMMD(model, model_arch = args.model_arch)
    elif args.algo == 'cot':
        algo = FinetuneCOT(model, model_arch = args.model_arch, data_loader = train_loader)
    elif args.algo == 'mars_pgm':
        algo = FinetuneMARSPGM(model, model_arch = args.model_arch)
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
        logits, features = model(image)

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
    accuracies = []
    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            label = paddle.unsqueeze(data[1], 1)
            logits, _ = model(image)

            loss = criterion(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

            avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

            if logger and batch_id % debug_steps == 0 and batch_id != 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {avg_loss}, " +
                    f"Avg Acc@1: {avg_acc}, ")

    val_time = time.time() - time_st
    return avg_loss, avg_acc, val_time


def finetune_cnn(args):
    # STEP 0: Preparation
    
    last_epoch = -1
    paddle.device.set_device(f'gpu:{args.gpu}')
    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    logger = get_logger(filename=os.path.join(args.save, f'{args.name}_{args.algo}.txt'))
    logdir = os.path.join(args.log_dir,args.algo)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = LogWriter(logdir = logdir)
    logger.info(f'\n{args}')

    # STEP 1: Create train and val dataloader
    dataloader_train, num_classes = get_dataloader_train(args)
    if os.path.exists(args.eval_dir):
        dataloader_val = get_dataloader_val(args)

    # STEP 2: load model
    model = eval(args.model_arch)(pretrained=True, num_classes = num_classes)
    logger.info('finish load the pretrained model')

    # STEP 3: determine algorithm for finetune
    algo = determine_algo(model, args, dataloader_train)

    # STEP 4: Define optimizer and lr_scheduler
    criterion = paddle.nn.CrossEntropyLoss()
    params = algo.params()
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[int(2.0*args.epochs/3.0)],values=[args.lr,args.lr*0.1])
    optimizer = paddle.optimizer.Momentum(learning_rate=lr_scheduler,parameters=params,momentum=0.9,use_nesterov=True,weight_decay = args.wd)

    # STEP 5: Run training
    logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, args.epochs+1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_acc, train_time = train(dataloader=dataloader_train,
                                                  model=model,
                                                  criterion=criterion,
                                                  algo=algo,
                                                  optimizer=optimizer,
                                                  epoch=epoch,
                                                  total_epochs=args.epochs,
                                                  total_batch=len(dataloader_train),
                                                  debug_steps=args.print_frequency,
                                                  logger=logger)
        lr_scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{args.epochs:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")
        writer.add_scalar(tag="train_acc", step=epoch, value=train_acc)
        writer.add_scalar(tag="train_loss", step=epoch, value=train_loss)

        # validation
        if os.path.exists(args.eval_dir):
            if epoch % args.eval_frequency == 0 or epoch == args.epochs:
                logger.info(f'----- Validation after Epoch: {epoch}')
                val_loss, val_acc, val_time = validate(
                    dataloader=dataloader_val,
                    model=model,
                    criterion=criterion,
                    total_batch=len(dataloader_val),
                    debug_steps=args.print_frequency,
                    logger=logger)
                logger.info(f"----- Epoch[{epoch:03d}/{args.epochs:03d}], " +
                            f"Validation Loss: {val_loss:.4f}, " +
                            f"Validation Acc@1: {val_acc:.4f}, " +
                            f"time: {val_time:.2f}")
                writer.add_scalar(tag="val_acc", step=epoch, value=val_acc)
                writer.add_scalar(tag="val_loss", step=epoch, value=val_loss)

        if epoch % args.save_frequency == 0 or epoch == args.epochs:
            model_path = os.path.join(args.save, f"{args.name}_{args.algo}_Epoch{epoch}.pdparams")
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
    finetune_cnn(args)
    
