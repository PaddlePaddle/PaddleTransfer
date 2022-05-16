from finetune_cnn import finetune_cnn
from finetune_vit import finetune_vit

import argparse

def get_args():
    parser = argparse.ArgumentParser(description = 'PaddlePaddle Deep Transfer Learning Toolkit, Image Classification Fine-tuning Example')
    # general configurations
    parser.add_argument('--name', type = str, default = 'dtd')
    parser.add_argument('--train_dir', default = '../CoTuning/data/finetune/dtd/train')
    parser.add_argument('--eval_dir', default = '../CoTuning/data/finetune/dtd/test')
    parser.add_argument('--log_dir', default = './visual_log')
    parser.add_argument('--save', type = str, default = './output')
    

    # different default configurations for different model arch
    parser.add_argument('--model_arch', default = 'resnet50')
    parser.add_argument('--epochs', type = int, default = None)
    parser.add_argument('--image_size', type = int, default = None)
    parser.add_argument('--batch_size', type = int, default = None)
    parser.add_argument('--batch_size_eval', type=int, default=None)
    parser.add_argument('--lr', type = float, default = None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--cfg', type = str, default = "./configs/vit_base_patch16_384.yaml")
    parser.add_argument('--model_path',default = './models/vit_base_patch16_384.pdparams')

    # training related configurations
    parser.add_argument('--algo', type=str, default = 'base')
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--print_frequency', type=int, default=50)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--save_frequency',default = 10)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args

def custom_args(args):
    if args.model_arch == 'vit':
        if not args.epochs:
            args.epochs = 64
        if not args.image_size:
            args.image_size = 384
        if not args.batch_size:
            args.batch_size = 8
        if not args.batch_size_eval:
            args.batch_size_eval = 8
        if not args.lr:
            args.lr = 1e-3
        finetune_vit(args)
    else:
        if not args.epochs:
            args.epochs = 40
        if not args.image_size:
            args.image_size = 224
        if not args.batch_size:
            args.batch_size = 64
        if not args.batch_size_eval:
            args.batch_size_eval = 32
        if not args.lr:
            args.lr = 1e-2
        if not args.wd:
            args.wd = 1e-4
        finetune_cnn(args)

def main():
    args = get_args()
    custom_args(args)
    

if __name__ == "__main__":
    main()