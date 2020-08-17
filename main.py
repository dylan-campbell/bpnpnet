# BLINDPNP SOLVER WITH DECLARATIVE SINKHORN AND PNP NODES
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Liu Liu <liu.liu@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#
# Modified from PyTorch ImageNet example:
# https://github.com/pytorch/examples/blob/ee964a2eeb41e1712fe719b83645c79bcbd0ba1a/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import cv2
import math
import pickle
import statistics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard as tb

from model.model import DBPnP
# from model.model_hungarian import DBPnP # At test time only
from lib.losses import *
import utilities.geometry_utilities as geo
from utilities.dataset_utilities import Dataset

# torch.manual_seed(2809)

parser = argparse.ArgumentParser(description='PyTorch DeepBlindPnP Training')
parser.add_argument('data_dir', metavar='DIR',
                    help='path to datasets directory')
parser.add_argument('--dataset', dest='dataset', default='', type=str,
                    help='dataset name')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--log-dir', dest='log_dir', default='', type=str,
                    help='Directory for logging loss and accuracy')
parser.add_argument('--num_points_train', default=1000, type=int, metavar='NTRAIN',
                    help='number of points for each training point-set (default: 1000)')
parser.add_argument('--poseloss', dest='poseloss', default=0, type=int,
                    help='specify epoch at which to introduce pose loss')
parser.add_argument('--frac_outliers', default=0.0, type=float,dest='frac_outliers')

def get_dataset(args):
    train_dataset = Dataset('train', args, args.batch_size, preprocessed=True)
    val_dataset   = Dataset('valid', args, 1, preprocessed=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, drop_last=True,
        collate_fn=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=args.workers, drop_last=True,
        collate_fn=None)
    return train_loader, val_loader

def main():
    best_error = float("inf")
    args = parser.parse_args()
    args.writer = tb.SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    model = DBPnP(args)

    if args.gpu is not None:
        print("Using GPU {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    if args.poseloss == 0:
        gamma = 1.0
    else:
        gamma = 0.0
    criterion = TotalLoss(gamma).cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay
                                 )

    # Optionally resume from a checkpoint
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
            best_error = checkpoint['best_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader, val_loader = get_dataset(args)

    if args.evaluate:
        test(val_loader, model, criterion, 0, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        if (args.poseloss <= epoch):
            criterion.gamma = 1.0

        train(train_loader, model, criterion, optimizer, epoch, args)
        error = validate(val_loader, model, criterion, epoch, args)

        is_best = error < best_error
        best_error = min(error, best_error)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'optimizer' : optimizer.state_dict(),
        }, is_best, dir=args.log_dir, filename='checkpoint_epoch_' + str(epoch + 1))

    if args.writer:
        args.writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')
    loss_meter = AverageMeter('Loss', ':6.4f')
    correspondence_probability_meter = AverageMeter('Outlier-Inlier Prob', ':6.4f')
    rotation_meter = AverageMeter('Rotation Error', ':6.4f')
    translation_meter = AverageMeter('Translation Error', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_meter, correspondence_probability_meter, rotation_meter, translation_meter],
        prefix="Epoch: [{}]".format(epoch))

    poseloss = (args.poseloss <= epoch)

    model.train()

    end = time.time()
    for batch_index, (p2d, p3d, R_gt, t_gt, C_gt, num_points_2d, num_points_3d) in enumerate(train_loader):
        data_time.update(time.time() - end) # Measure data loading time

        p2d = p2d.float()
        p3d = p3d.float()
        R_gt = R_gt.float()
        t_gt = t_gt.float()

        if args.gpu is not None:
            p2d = p2d.cuda(args.gpu, non_blocking=True)
            p3d = p3d.cuda(args.gpu, non_blocking=True)
        R_gt = R_gt.cuda(args.gpu, non_blocking=True)
        t_gt = t_gt.cuda(args.gpu, non_blocking=True)
        C_gt = C_gt.cuda(args.gpu, non_blocking=True)

        # Convert C_gt into matrix (stored as a b x n index tensor with outliers indexed by m)
        m = p2d.size(-2)
        C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()

        # Handle duplicates by setting them all to 0:
        for i in range(C_gt.size(0)):
            C_gt[i, num_points_2d[i]:, :] = 0.0
            C_gt[i, :, num_points_3d[i]:] = 0.0

        # Compute output
        P, theta, theta0 = model(p2d, p3d, num_points_2d, num_points_3d, poseloss)

        loss, losses = criterion(theta, P, R_gt, t_gt, C_gt)
        loss_correspondence_probability = losses[0]
        loss_meter.update(loss.item(), p2d.size(0))
        correspondence_probability_meter.update(loss_correspondence_probability.item(), p2d.size(0))
        if poseloss:
            loss_rotation = losses[1]
            loss_translation = losses[2]
            rotation_meter.update(loss_rotation.item(), p2d.size(0))
            translation_meter.update(loss_translation.item(), p2d.size(0))

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.writer:
            global_step = epoch * len(train_loader) + batch_index
            args.writer.add_scalar('loss_train', loss.item(), global_step=global_step)
            args.writer.add_scalar('correspondence_probability_train', loss_correspondence_probability.item(), global_step=global_step)
            if poseloss:
                args.writer.add_scalar('rotation_train', loss_rotation.item() * 180.0 / math.pi, global_step=global_step)
                args.writer.add_scalar('translation_train', loss_translation.item(), global_step=global_step)

        if batch_index % args.print_freq == 0:
            progress.display(batch_index)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':6.4f')
    correspondence_probability_meter = AverageMeter('Outlier-Inlier Prob', ':6.4f')
    rotation_meter = AverageMeter('Rotation Error', ':6.4f')
    translation_meter = AverageMeter('Translation Error', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, loss_meter, correspondence_probability_meter, rotation_meter, translation_meter],
        prefix='Test: ')

    poseloss = (args.poseloss <= epoch)

    model.eval()

    with torch.no_grad():
        end = time.time()

        rotation_errors_theta0 = []
        translation_errors_theta0 = []

        start_time = time.time()
        for batch_index, (p2d, p3d, R_gt, t_gt, C_gt, num_points_2d, num_points_3d) in enumerate(val_loader):
            p2d = p2d.float()
            p3d = p3d.float()
            R_gt = R_gt.float()
            t_gt = t_gt.float()
            if args.gpu is not None:
                p2d = p2d.cuda(args.gpu, non_blocking=True)
                p3d = p3d.cuda(args.gpu, non_blocking=True)
            R_gt = R_gt.cuda(args.gpu, non_blocking=True)
            t_gt = t_gt.cuda(args.gpu, non_blocking=True)
            C_gt = C_gt.cuda(args.gpu, non_blocking=True)

            # Convert C_gt into matrix (stored as a b x n index tensor with outliers indexed by m)
            m = p2d.size(-2)
            C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()

            # Compute output
            P, theta, theta0 = model(p2d, p3d, num_points_2d, num_points_3d, poseloss)

            # Measure elapsed time for reporting (includes dataloading time, but not loss / error measure computation time)
            inference_time = (time.time() - start_time)

            loss, losses = criterion(theta, P, R_gt, t_gt, C_gt)
            loss_correspondence_probability = losses[0]
            loss_meter.update(loss.item(), p2d.size(0))
            correspondence_probability_meter.update(loss_correspondence_probability.item(), p2d.size(0))
            if poseloss:
                loss_rotation = losses[1]
                loss_translation = losses[2]
                rotation_meter.update(loss_rotation.item(), p2d.size(0))
                translation_meter.update(loss_translation.item(), p2d.size(0))
                rotation_errors_theta0 += [rotationErrorsTheta(theta0, R_gt).item()]
                translation_errors_theta0 += [translationErrorsTheta(theta0, t_gt).item()]

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            start_time = time.time()

            if batch_index % args.print_freq == 0:
                progress.display(batch_index)

        print('Loss: {loss.avg:6.4f}, Outlier-Inlier Prob: {correspondence_probability.avg:6.4f}, Rotation Error: {rot.avg:6.4f}, Translation Error: {transl.avg:6.4f}'
              .format(loss=loss_meter, correspondence_probability=correspondence_probability_meter, rot=rotation_meter, transl=translation_meter))

        if args.writer:
            args.writer.add_scalar('loss_val', loss_meter.avg, global_step=epoch)
            args.writer.add_scalar('correspondence_probability_val', correspondence_probability_meter.avg, global_step=epoch)
            if poseloss:
                args.writer.add_scalar('rotation_val', rotation_meter.avg * 180.0 / math.pi, global_step=epoch)
                args.writer.add_scalar('translation_val', translation_meter.avg, global_step=epoch)
                args.writer.add_scalar('rotation_median_val', statistics.median(rotation_errors_theta0) * 180.0 / math.pi, global_step=epoch)
                args.writer.add_scalar('translation_median_val', statistics.median(translation_errors_theta0), global_step=epoch)
    return loss_meter.avg

def test(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':6.4f')
    correspondence_probability_meter = AverageMeter('Outlier-Inlier Prob', ':6.4f')
    rotation_meter = AverageMeter('Rotation Error', ':6.4f')
    translation_meter = AverageMeter('Translation Error', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, loss_meter, correspondence_probability_meter, rotation_meter, translation_meter],
        prefix='Test: ')

    poseloss = True
    criterion.gamma = 1.0

    model.eval()

    with torch.no_grad():
        end = time.time()

        rotation_errors0, rotation_errors, rotation_errorsLM = [], [], []
        translation_errors0, translation_errors, translation_errorsLM = [], [], []
        reprojection_errors0, reprojection_errors, reprojection_errorsLM, reprojection_errorsGT = [], [], [], []
        num_inliers0, num_inliers, num_inliersLM, num_inliersGT = [], [], [], []
        thetas0, thetas, thetasLM = [], [], []
        num_points_2d_list, num_points_3d_list = [], []
        inference_times = []

        start_time = time.time()
        for batch_index, (p2d, p3d, R_gt, t_gt, C_gt, num_points_2d, num_points_3d) in enumerate(val_loader):

            p2d = p2d.float()
            p3d = p3d.float()
            R_gt = R_gt.float()
            t_gt = t_gt.float()

            if args.frac_outliers == 0.0:
                # Convert C_gt into matrix (stored as a b x n index tensor with outliers indexed by m)
                m = p2d.size(-2)
                C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()
            elif args.frac_outliers > 0.0:
                # Add random outliers:
                # Get bounding boxes
                bb2d_min = p2d.min(dim=-2)[0]
                bb2d_width = p2d.max(dim=-2)[0] - bb2d_min
                bb3d_min = p3d.min(dim=-2)[0]
                bb3d_width = p3d.max(dim=-2)[0] - bb3d_min
                num_outliers = int(args.frac_outliers * p2d.size(-2))
                p2d_outliers = bb2d_width * torch.rand_like(p2d[:, :num_outliers, :]) + bb2d_min
                p3d_outliers = bb3d_width * torch.rand_like(p3d[:, :num_outliers, :]) + bb3d_min
                p2d = torch.cat((p2d, p2d_outliers), -2)
                p3d = torch.cat((p3d, p3d_outliers), -2)
                num_points_2d[0, 0] = p2d.size(-2)
                num_points_3d[0, 0] = p3d.size(-2)
                # Expand C_gt with outlier indices (b x n index tensor with outliers indexed by m)
                b = p2d.size(0)
                m = p2d.size(-2)
                outlier_indices = C_gt.new_full((b, num_outliers), m)
                C_gt = torch.cat((C_gt, outlier_indices), -1)
                C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()
                # For memory reasons, if num_points > 10000, downsample first
                if p2d.size(-2) > 10000:
                    idx = torch.randint(p2d.size(-2), size=(10000,))
                    p2d = p2d[:, idx, :]
                    p3d = p3d[:, idx, :]
                    num_points_2d[0, 0] = p2d.size(-2)
                    num_points_3d[0, 0] = p3d.size(-2)
                    C_gt = C_gt[:, idx, :]
                    C_gt = C_gt[:, :, idx]

            if args.gpu is not None:
                p2d = p2d.cuda(args.gpu, non_blocking=True)
                p3d = p3d.cuda(args.gpu, non_blocking=True)
            R_gt = R_gt.cuda(args.gpu, non_blocking=True)
            t_gt = t_gt.cuda(args.gpu, non_blocking=True)
            C_gt = C_gt.cuda(args.gpu, non_blocking=True)

            # Compute output
            P, theta, theta0 = model(p2d, p3d, num_points_2d, num_points_3d, poseloss)

            # Measure elapsed time for reporting (includes dataloading time, but not evaluation / loss time)
            inference_time = (time.time() - start_time)

            loss, losses = criterion(theta, P, R_gt, t_gt, C_gt)
            loss_correspondence_probability = losses[0]
            loss_meter.update(loss.item(), p2d.size(0))
            correspondence_probability_meter.update(loss_correspondence_probability.item(), p2d.size(0))
            if poseloss:    
                loss_rotation = losses[1]
                loss_translation = losses[2]
                rotation_meter.update(loss_rotation.item(), p2d.size(0))
                translation_meter.update(loss_translation.item(), p2d.size(0))

            # Compute refined pose estimate:
            # 1. Find inliers based on RANSAC estimate
            inlier_threshold = 1.0 * math.pi / 180.0 # 1 degree threshold for LM
            C = correspondenceMatricesTheta(theta0, p2d, p3d, inlier_threshold)
            K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
            dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
            thetaLM = P.new_zeros((P.size(0), 6))
            inlier_indices = C[0, ...].nonzero(as_tuple=True) # Assumes test batch size = 1
            # Skip if point-set has < 4 inlier points:
            if (inlier_indices[0].size()[0] >= 4):
                p2d_np = p2d[0, inlier_indices[0], :].cpu().numpy()
                p3d_np = p3d[0, inlier_indices[1], :].cpu().numpy()
                rvec = theta0[0, :3].cpu().numpy()
                tvec = theta0[0, 3:].cpu().numpy()
                rvec, tvec = cv2.solvePnPRefineLM(p3d_np, p2d_np, K, dist_coeff, rvec, tvec)
                if rvec is not None and tvec is not None:
                    thetaLM[0, :3] = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
                    thetaLM[0, 3:] = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)

            inlier_threshold = 0.1 * math.pi / 180.0 # 0.1 degree threshold for reported inlier count
            rotation_errors0 += [rotationErrorsTheta(theta0, R_gt, eps=0.0).item()]
            rotation_errors += [rotationErrorsTheta(theta, R_gt, eps=0.0).item()]
            rotation_errorsLM += [rotationErrorsTheta(thetaLM, R_gt, eps=0.0).item()]
            translation_errors0 += [translationErrorsTheta(theta0, t_gt).item()]
            translation_errors += [translationErrorsTheta(theta, t_gt).item()]
            translation_errorsLM += [translationErrorsTheta(thetaLM, t_gt).item()]
            reprojection_errors0 += [reprojectionErrorsTheta(theta0, p2d, p3d, C_gt, eps=0.0).item()]
            reprojection_errors += [reprojectionErrorsTheta(theta, p2d, p3d, C_gt, eps=0.0).item()]
            reprojection_errorsLM += [reprojectionErrorsTheta(thetaLM, p2d, p3d, C_gt, eps=0.0).item()]
            reprojection_errorsGT += [reprojectionErrors(R_gt, t_gt, p2d, p3d, C_gt, eps=0.0).item()]
            num_inliers0 += [numInliersTheta(theta0, p2d, p3d, inlier_threshold).item()]
            num_inliers += [numInliersTheta(theta, p2d, p3d, inlier_threshold).item()]
            num_inliersLM += [numInliersTheta(thetaLM, p2d, p3d, inlier_threshold).item()]
            num_inliersGT += [numInliers(R_gt, t_gt, p2d, p3d, inlier_threshold).item()]
            num_points_2d_list += [num_points_2d[0].item()]
            num_points_3d_list += [num_points_3d[0].item()]
            inference_times += [inference_time]
            thetas0 += [theta0[0, 0].item(), theta0[0, 1].item(), theta0[0, 2].item(), theta0[0, 3].item(), theta0[0, 4].item(), theta0[0, 5].item()]
            thetas += [theta[0, 0].item(), theta[0, 1].item(), theta[0, 2].item(), theta[0, 3].item(), theta[0, 4].item(), theta[0, 5].item()]
            thetasLM += [thetaLM[0, 0].item(), thetaLM[0, 1].item(), thetaLM[0, 2].item(), thetaLM[0, 3].item(), thetaLM[0, 4].item(), thetaLM[0, 5].item()]

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            start_time = time.time()

            if batch_index % args.print_freq == 0:
                progress.display(batch_index)

        print('Loss: {loss.avg:6.4f}, Outlier-Inlier Prob: {correspondence_probability.avg:6.4f}, Rotation Error: {rot.avg:6.4f}, Translation Error: {transl.avg:6.4f}'
              .format(loss=loss_meter, correspondence_probability=correspondence_probability_meter, rot=rotation_meter, transl=translation_meter))

        # Print all results in a text file:
        if args.poseloss == 0:
            loss_string = 'LcLp'
        else:
            loss_string = 'Lc'
        if args.frac_outliers == 0.0:
            append_string = ''
        elif args.frac_outliers > 0.0:
            append_string = '_outliers_' + str(args.frac_outliers)
        dataset_string = args.dataset
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./results/' + dataset_string, exist_ok=True)
        os.makedirs('./results/' + dataset_string + '/' + loss_string, exist_ok=True)
        with open('./results/' + dataset_string + '/' + loss_string + '/results' + append_string + '.txt', 'w') as save_file:
            save_file.write("rotation_errors0, rotation_errors, rotation_errorsLM, translation_errors0, translation_errors, translation_errorsLM, reprojection_errors0, reprojection_errors, reprojection_errorsLM, reprojection_errorsGT, num_inliers0, num_inliers, num_inliersLM, num_inliersGT, num_points_2d, num_points_3d, inference_time\n")
            for i in range(len(rotation_errors)):
                save_file.write("{:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {}, {}, {}, {}, {}, {}, {:.9f}\n".format(
                    rotation_errors0[i], rotation_errors[i], rotation_errorsLM[i],
                    translation_errors0[i], translation_errors[i], translation_errorsLM[i],
                    reprojection_errors0[i], reprojection_errors[i], reprojection_errorsLM[i], reprojection_errorsGT[i],
                    num_inliers0[i], num_inliers[i], num_inliersLM[i], num_inliersGT[i],
                    num_points_2d_list[i], num_points_3d_list[i],
                    inference_times[i]
                    ))
        with open('./results/' + dataset_string + '/' + loss_string + '/poses' + append_string + '.txt', 'w') as save_file:
            for i in range(len(rotation_errors)):
                save_file.write("{:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}\n".format(
                    thetas0[6*i + 0], thetas0[6*i + 1], thetas0[6*i + 2], thetas0[6*i + 3], thetas0[6*i + 4], thetas0[6*i + 5],
                    thetas[6*i + 0], thetas[6*i + 1], thetas[6*i + 2], thetas[6*i + 3], thetas[6*i + 4], thetas[6*i + 5],
                    thetasLM[6*i + 0], thetasLM[6*i + 1], thetasLM[6*i + 2], thetasLM[6*i + 3], thetasLM[6*i + 4], thetasLM[6*i + 5]
                    ))
        # Pickle all output camera poses:
        poses = {}
        poses["theta0"] = np.array(thetas0).reshape(-1, 6)
        poses["theta"] = np.array(thetas).reshape(-1, 6)
        poses["thetaLM"] = np.array(thetasLM).reshape(-1, 6)
        pickle.dump(poses, open('./results/' + dataset_string + '/' + loss_string + '/poses' + append_string + '.pkl', 'wb'))
    return loss_meter.avg

def save_checkpoint(state, is_best, dir='', filename='checkpoint'):
    torch.save(state, dir + filename + '.pth.tar')
    if is_best:
        shutil.copyfile(dir + filename + '.pth.tar', dir + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
