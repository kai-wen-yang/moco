#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from moco import models
import moco.loader
import moco.builder_morph
import pdb
import faiss
from utils import *
from vtabdataset import VTAB
import numpy
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from main_lincls import validate
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# options for morphing
parser.add_argument('--target_dataset', default='caltech101', type=str,
                    help='target dataset name')
parser.add_argument('--target_data_path', default='../data/vtab-1k', type=str)
parser.add_argument('--train_iter', default=24, type=int,
                    help='number of training iterations per epoch')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_cluster', default=2000, type=int)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.num_target_classes = DATA2CLS[args.target_dataset]
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder_morph.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.num_target_classes)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                new_state_dict[name] = v

            msg = model.load_state_dict(new_state_dict, strict=False)
            print(msg)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = moco.loader.ImageFolderInstance(
        traindir,
        transforms.Compose(augmentation))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    print("=> data loaded")
    # target dataset
    data_path = os.path.join(args.target_data_path, args.target_dataset)
    target_dataset = moco.loader.VTAB(root=data_path, train=True, transform=transforms.Compose(augmentation))
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    data_indices = np.arange(len(train_dataset))
    centroids = torch.zeros(int(args. num_cluster), args.moco_dim).cuda()
    im2cluster = torch.zeros(len(train_dataset), dtype=torch.long).cuda()
    #data_indices, centroids, im2cluster = morphing(train_loader, target_loader, model, data_indices, args)

    if args.gpu==0:
        #features = torch.load('features2.pt').cpu()
        #features = features.numpy()
        # cluster_result = run_kmeans(features, args)  # run kmeans clustering on master node
        # torch.save(cluster_result, './cluster')
        cluster_result = torch.load('cluster')
        im2cluster = cluster_result['im2cluster'][0] # select length
        centroids = cluster_result['centroids'][0] # select length, 128

        target_features = compute_target_features(target_loader, model, args)  # num target class, 128
        similarity = F.normalize(centroids).mm(F.normalize(target_features).t())  # cluster num, class num
        similarity_sum = torch.sum(similarity, dim=1)  # cluster num
        sorted, indices = torch.sort(similarity_sum)
        selected_indices = indices[:int(args.num_cluster * 0.1)]  # cluster num
        select = torch.Tensor(
            [1 if item in selected_indices else 0 for item in im2cluster.tolist()]).bool()  # select length, 1
        data_indices = data_indices[select]  # new length
        im2cluster = im2cluster[data_indices]

    if args.distributed:
        dist.barrier()
        dist.broadcast(centroids, 0, async_op=False)
        dist.broadcast(im2cluster, 0, async_op=False)

    train_loader = torch.utils.data.DataLoader(
        moco.loader.SubSetInstance(train_dataset, data_indices), batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, target_loader, model, criterion, optimizer, epoch, centroids, im2cluster, args)
        acc1 = validate(val_loader, model, criterion, args)
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def morphing(train_loader, target_loader, model, data_indices, args):
    features = compute_features(train_loader, model, args) # select length, 128
    if args.gpu==0:
        torch.save(features, './features2.pt')

        features = features.numpy()
        cluster_result = run_kmeans(features, args)  # run kmeans clustering on master node
        im2cluster = cluster_result['im2cluster'][0] # select length
        centroids = cluster_result['centroids'][0] # select length, 128
        torch.save(cluster_result, './cluster_result')

    dist.barrier()
    target_features = compute_target_features(target_loader, model, args) # num target class, 128

    similarity = F.normalize(centroids).mm(F.normalize(target_features).t()) # cluster num, class num
    similarity_sum = torch.sum(similarity, dim=1) # cluster num
    sorted, indices = torch.sort(similarity_sum)
    selected_indices = indices[:args.num_cluster*0.3] # cluster num
    select = torch.Tensor([1 if item in selected_indices else 0 for item in im2cluster.tolist()]).bool() # select length, 1
    data_indices = data_indices[select] # new length

    return data_indices, centroids, im2cluster[select]


def train(train_loader, target_loader, model, criterion, optimizer, epoch, centroids, im2cluster, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.train_iter,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_i in range(args.train_iter):
        try:
            inputs_p, index_p = train_loader_iter.next()
        except:
            train_loader_iter = iter(train_loader)
            inputs_p, index_p = train_loader_iter.next()

        try:
            inputs_t, label_t = target_loader_iter.next()
        except:
            target_loader_iter = iter(target_loader)
            inputs_t, label_t = target_loader_iter.next()
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            inputs_p = inputs_p.cuda(args.gpu, non_blocking=True)
            inputs_t = inputs_t.cuda(args.gpu, non_blocking=True)
            index_p = index_p.cuda(args.gpu, non_blocking=True)
            label_t = label_t.cuda(args.gpu, non_blocking=True)

        # compute output
        output_t = model(im_q=inputs_t, is_target=True)
        output_p, target_p = model(im_q=inputs_p, centroids=centroids, im2cluster=im2cluster, index=index_p)
        loss = criterion(output_p, target_p)+criterion(output_t, label_t)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_t, label_t, topk=(1, 5))
        losses.update(loss.item(), inputs_p[0].size(0))
        top1.update(acc1[0], inputs_p[0].size(0))
        top5.update(acc5[0], inputs_p[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_i % args.print_freq == 0:
            progress.display(batch_i)


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.moco_dim).cuda()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, data_time])

    end = time.time()
    for i, (images, index) in enumerate(eval_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images, is_eval=True)
            features[index] = feat
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    if args.distributed:
        dist.barrier()
        dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


def compute_target_features(eval_loader, model, args):
    print('Computing target features...')
    model.eval()
    features = torch.zeros(args.num_target_classes, args.moco_dim).cuda()
    for c in range(args.num_target_classes):
        print(c)
        class_index = np.arange(len(eval_loader.dataset))[eval_loader.dataset.targets == c]
        temp_features = torch.zeros(len(class_index), args.moco_dim).cuda()
        temp_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(eval_loader.dataset, class_index),
            batch_size=len(class_index), num_workers=args.workers, pin_memory=True)

        for images, _ in temp_loader:
            with torch.no_grad():
                images = images.cuda(non_blocking=True)
                feat = model(images, is_eval=True)
                temp_features = feat # num_per_class, 128
        features[c, :] = torch.mean(temp_features, dim=0) # num_class 128

        if args.distributed:
            dist.barrier()

    return features


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    num_cluster = args.num_cluster
    # intialize faiss clustering parameters
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = args.gpu
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]

    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    # sample-to-centroid distances for each cluster
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    # concentration estimation (phi)
    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    density = density / density.mean()  # scale the mean to temperature

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)

    return results


if __name__ == '__main__':
    main()