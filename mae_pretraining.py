# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import sys
import math
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from datasets import *
from models import *
from utils import *
from knntest import *


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    # dataset: 4_32, 7_84, 8_96
    parser.add_argument('--model', default='vit_7_84', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--cos', default=True, type=bool, help='Use cosine scheduler')

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--dataset', default='miniImageNet', type=str,
                        help='dataset name')                 
    parser.add_argument('--output_dir', default='logs_noaug_mini', type=str,
                        help='path where to save, empty for no saving')

    return parser.parse_args()


def main(args):
    # print(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # fix the seed for reproducibility
    torch.manual_seed(546)
    np.random.seed(546)

    cudnn.benchmark = True

    if 'vit' in args.model:
        model = VIT(args).to(device)
    elif 'swin' in args.model:
        model = SWIN(args).to(device)
    else:
        print("unsupported model:", args.model)
        sys.exit(1)

    args.input_size = int(args.model.split("_")[-1])
    
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    train_dataset, test_dataset = build_dataset(args)
    # unlabeled_stl10 = STL10(root='stl10', split='unlabeled', transform=DataAugmentationForMAE(args, True), download=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=True,
    )
    # unlabeled_stl10_dataloader = torch.utils.data.DataLoader(
    #     unlabeled_stl10,
    #     batch_size=args.batch_size,
    #     num_workers=16,
    #     drop_last=True,
    # )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=True,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params: {} M'.format(n_parameters / 1e6))

    optimizer = create_optimizer(args, model)
    
    loss_func = nn.MSELoss()
    normlize_target = True

    training_loss_list = []
    accuracy_list = []

    ep = 0
    # mini training down
    resume = False
    if resume:
        ckpt = torch.load('./logs_noaug_cifar10/train_epoch_179.pth')
        ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        training_loss_list = ckpt['train_loss_list']

    print(f"Start training for {args.epochs} epochs")
    model.train()
    for epoch in range(ep+1, args.epochs):
        for i, (batch,_) in enumerate(train_dataloader):
            images, bool_masked_pos = batch
            images = images.to(device)
            bool_masked_pos = bool_masked_pos.to(device).flatten(1).to(torch.bool)
            
            with torch.no_grad():
                # calculate the predict label
                if args.dataset == 'cifar10':
                    mean = torch.as_tensor((0.4914, 0.4822, 0.4465)).to(device)[None, :, None, None]
                    std = torch.as_tensor((0.2023, 0.1994, 0.2010)).to(device)[None, :, None, None]
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
                unnorm_images = images * std + mean  # in [0, 1]

                if normlize_target:
                    images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
                    images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                        ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[0])

                B, _, C = images_patch.shape
                labels = images_patch[bool_masked_pos].reshape(B, -1, C)
            

            outputs, encoding = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

            loss_value = loss.item()
            training_loss_list.append(loss_value)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("epoch:", epoch, "train loss:", loss_value)

        # knn test. For stl10-unlabeled: train using unlabeled, test using labeled dataloaders
        test_acc = test(model.encoder, train_dataloader, test_dataloader, epoch, args)
        
        # adjust lr
        adjust_learning_rate(optimizer, epoch, args)

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                path = 'logs_noaug_mini/train_epoch_' + str(epoch) + '.pth'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test acc': test_acc,
                    'train_loss_list': training_loss_list
                }, path)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
