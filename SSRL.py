

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('SSRL', add_help=False)

    ##======== System ============
    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        help="""Random seed.""")
    parser.add_argument("--num_workers",
                        default=12,
                        type=int,
                        help="""Number of data loading workers per GPU.""")
    parser.add_argument("--use_prefetcher",
                        type=utils.bool_flag,
                        default=True,
                        help="""whether we use prefetcher which can accerelate the training speed.""")

    ##======== Path ============
    parser.add_argument("--data_path",
                        default='E:\kaggle/train',
                        type=str,
                        help="""Path to the ImageNet training data.""")
    parser.add_argument("--output_dir",
                        default="D:\SSRL\check_train",
                        type=str,
                        help="""Path to save logs and checkpoints.""")
    parser.add_argument("--saveckp_freq",
                        default=5,
                        type=int,
                        help="""Save checkpoint every x epochs.""")

    ##======== Model parameters ============
    parser.add_argument("--arch",
                        default='vit_small',
                        type=str,
                        choices=['vit_small', 'vit_base'] + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Architecture to train. For quick experiments, use vit_tiny or vit_small.""")
    parser.add_argument("--patch_size",
                        default=8,
                        type=int,
                        help="""Size of input square patches.""")
    parser.add_argument('--out_dim',
                        default=65536,
                        type=int,
                        help="""the queue size of the memory to store the negative keys.""")
    parser.add_argument("--drop_path_rate",
                        type=float,
                        default=0.1,
                        help="""stochastic depth rate""")

    ##======== Training/Optimization parameters ============
    parser.add_argument("--momentum_teacher",
                        default=0.99,
                        type=float,
                        help="""Base EMA
                        parameter for teacher update. The value is increased to 1 during training with
                        cosine schedule. We recommend setting a higher value with small batches: for
                        example use 0.9995 with batch size of 256.""")
    parser.add_argument("--norm_last_layer",
                        default=False,
                        type=utils.bool_flag,
                        help="""Whether to weight normalize the last layer.""")
    parser.add_argument("--use_fp16",
                        type=utils.bool_flag,
                        default=False,
                        help="""Whether or not
                        to use half precision for training. Improves training time and memory requirements,
                        but can provoke instability and slight decay of performance. We recommend disabling
                        mixed precision if the loss is unstable, if reducing the patch size or if training
                        with bigger ViTs.""")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.04,
                        help="""Initial value of the
                        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument("--weight_decay_end",
                        type=float,
                        default=0.4,
                        help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")
    parser.add_argument('--use_bn_in_head',
                        default=False,
                        type=utils.bool_flag,
                        help="""Whether to use batch normalization in the projection head.""")
    parser.add_argument("--min_lr",
                        type=float,
                        default=1e-6,
                        help="""Target LR at the
                        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        help="""Learning rate at the end of
                        linear warmup (highest LR used during training). The learning rate is linearly scaled
                        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--clip_grad",
                        type=float,
                        default=3.0,
                        help="""Maximal parameter gradient norm.""")
    parser.add_argument("--batch_size_per_gpu",
                        default=20,
                        type=int,
                        help="""Per-GPU batch size.""")
    parser.add_argument("--epochs",
                        default=90,
                        type=int,
                        help="""Number of epochs of training.""")
    parser.add_argument("--warmup_epochs",
                        default=10,
                        type=int,
                        help="""Number of epochs for linear learning-rate warmup.""")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adamw",
                        choices=["adamw", "sgd", "lars"],
                        help="""Type of optimizer. We recommend using adamw
                        with ViTs.""")

    ##======== Temperature teacher parameters ============
    parser.add_argument('--warmup_teacher_temp',
                        default=0.04,
                        type=float,
                        help="Initial teacher temperature.")
    parser.add_argument('--teacher_temp',
                        default=0.04,
                        type=float,
                        help="Final teacher temperature.")
    parser.add_argument('--warmup_teacher_temp_epochs',
                        default=0,
                        type=int,
                        help="Number of warmup epochs for teacher temperature.")

    ##======== Crops parameters ============
    parser.add_argument('--global_crops_scale',
                        type=float,
                        nargs='+',
                        default=(0.4, 1.),
                        help="""Scale range for global crops.""")
    parser.add_argument('--local_crops_number',
                        type=int,
                        default=6,
                        help="""Number of small local views.""")
    parser.add_argument('--local_crops_scale',
                        type=float,
                        nargs='+',
                        default=(0.05, 0.4),
                        help="""Scale range for local crops.""")

    parser.add_argument("--dist_url",
                        default="env://", type=str, help="""url used to set up
           distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    return parser

def train_SSRL_model(args):

    # Initialize training environment
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    # Setup data loading
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Initialize networks
    student, teacher, teacher_without_ddp = create_networks(args)

    # Initialize loss, optimizer and schedulers
    SSRL_loss = create_loss(args)
    optimizer = create_optimizer(args, student)
    lr_schedule, wd_schedule, momentum_schedule = create_schedulers(args, data_loader)

    # Setup mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Resume from checkpoint if available
    start_epoch = resume_from_checkpoint(args, student, teacher, optimizer, fp16_scaler, SSRL_loss)

    # Main training loop
    print("Starting SSRL training!")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # Train one epoch
        train_stats = train_epoch(
            student=student,
            teacher=teacher,
            teacher_without_ddp=teacher_without_ddp,
            SSRL_loss=SSRL_loss,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            epoch=epoch,
            fp16_scaler=fp16_scaler,
            args=args
        )

        # Save checkpoint
        save_checkpoint(
            args=args,
            epoch=epoch,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            SSRL_loss=SSRL_loss,
            train_stats=train_stats
        )

    # Print total training time
    total_time = time.time() - start_time
    print(f'Training time: {datetime.timedelta(seconds=int(total_time))}')


def create_networks(args):
    """Creates and initializes student and teacher networks"""
    args.arch = args.arch.replace("deit", "vit")

    # Create base networks
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # Wrap networks
    student = utils.MultiCropWrapper(
        student,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, args.norm_last_layer)
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    )

    # Move to GPU and handle distributed training
    student, teacher = student.cuda(), teacher.cuda()

    # Handle batch norm synchronization
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # Initialize teacher with student weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # Freeze teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher, teacher_without_ddp


def create_loss(args):
    """Creates and initializes SSRL loss"""
    return SSRL_Loss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs
    ).cuda()


def create_optimizer(args, student):
    """Creates and initializes optimizer"""
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        return utils.LARS(params_groups)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_schedulers(args, data_loader):
    """Creates learning rate, weight decay, and momentum schedulers"""
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader)
    )

    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs,
        len(data_loader)
    )

    return lr_schedule, wd_schedule, momentum_schedule


def resume_from_checkpoint(args, student, teacher, optimizer, fp16_scaler, SSRL_loss):
    """Handles checkpoint loading and resume logic"""
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        SSRL_loss=SSRL_loss
    )
    return to_restore["epoch"]


def train_epoch(student, teacher, teacher_without_ddp, SSRL_loss, data_loader,
                optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                fp16_scaler, args):
    """Trains one epoch of SSRL"""
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Update learning rate and weight decay
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]

        # Forward passes and loss computation
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = SSRL_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # Teacher EMA update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(args, epoch, student, teacher, optimizer, fp16_scaler, SSRL_loss, train_stats):
    """Saves training checkpoint and logs"""
    save_dict = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'args': args,
        'SSRL_loss': SSRL_loss.state_dict(),
    }

    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()

    utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

    if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

class DataAugmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
class SSRL_Loss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSRL', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SSRL_model(args)