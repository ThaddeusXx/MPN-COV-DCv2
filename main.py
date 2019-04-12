import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np

import dataset_processing
from imgaug import augmenters as iaa
from dataset_processing import *
from torchvision import datasets
from functions import *
from imagepreprocess import *
from model_init import *
from src.representation import *
from src.cyclic_lr_scheduler import CyclicLR
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

DATA_PATH = '/workspace/mnt/group/other/luchao/sample_all/ducha'
TRAIN_DATA = 'CAT_V2'
TEST_DATA = 'CAT_V2'
TRAIN_IMG_FILE = 'CAT_V2/train.txt'
TEST_IMG_FILE = 'CAT_V2/val.txt'


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-method', default='step', type=str,
                    help='method of learning rate')
parser.add_argument('--lr-params', default=[], dest='lr_params',nargs='*',type=float,
                    action='append', help='params of lr method')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--modeldir', default=None, type=str,
                    help='director of checkpoint')
parser.add_argument('--representation', default=None, type=str,
                    help='define the representation method')
parser.add_argument('--num-classes', default=None, type=int,
                    help='define the number of classes')
parser.add_argument('--freezed-layer', default=None, type=int,
                    help='define the end of freezed layer')
parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
                    help='store checkpoint in every epoch')
parser.add_argument('--classifier-factor', default=None, type=int,
                    help='define the multiply factor of classifier')
parser.add_argument('--benchmark', default=None, type=str,
                    help='name of dataset')
parser.add_argument('--expand_num', default=3, type=int,
                    help='number of imgaug expand')
best_prec1 = 0

aug = iaa.SomeOf(2,[iaa.Add((-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.04*255)),
    iaa.Multiply((0.75, 1.25)),
    iaa.Grayscale((0.05, 0.1)),
#    iaa.Sequential([iaa.GammaContrast((0.81, 1.1))]),
    iaa.Fliplr(1.0)
])

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
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

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.representation == 'GAvP':
        representation = {'function':GAvP,
                          'input_dim':2048}
    elif args.representation == 'MPNCOV':
        representation = {'function':MPNCOV,
                          'iterNum':5,
                          'is_sqrt':True,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':None if args.pretrained else 256}
    elif args.representation == 'BCNN':
        representation = {'function':BCNN,
                          'is_vec':True,
                          'input_dim':2048}
    elif args.representation == 'CBP':
        representation = {'function':CBP,
                          'thresh':1e-8,
                          'projDim':8192,
                          'input_dim': 512}
    else:
        warnings.warn('=> You did not choose a global image representation method!')
        representation = None # which for original vgg or alexnet

    model = get_model(args.arch,
                      representation,
                      args.num_classes,
                      args.freezed_layer,
                      pretrained=args.pretrained)
    # plot network
    vizNet(model, args.modeldir)
    # obtain learning rate
    LR = Learning_rate_generater(args.lr_method, args.lr_params, args.epochs)
    if args.pretrained:
        params_list = [{'params': model.features.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
        params_list.append({'params': model.representation.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay})
        params_list.append({'params': model.classifier1.parameters(),
                            'lr': args.lr*args.classifier_factor,
                            'weight_decay': 0. if args.arch.startswith('vgg') else args.weight_decay})
        params_list.append({'params': model.classifier2.parameters(),
                            'lr': args.lr*args.classifier_factor,
                            'weight_decay': 0. if args.arch.startswith('vgg') else args.weight_decay})
    else:
        params_list = [{'params': model.features.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
        params_list.append({'params': model.representation.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay})
        params_list.append({'params': model.classifier.parameters(),
                            'lr': args.lr*args.classifier_factor,
                            'weight_decay':args.weight_decay})

    optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = CyclicLR(optimizer, base_lr=0.000001, max_lr=0.005, step_size=8, mode='exp_range')
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_dict = model.state_dict()
            ckp_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            #model.load_state_dict(checkpoint['state_dict'])
            model_dict.update(ckp_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(args.benchmark)
    '''train_dataset = dataset_processing.DatasetProcessing(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE,
                                                         TRAIN_LABEL_FILE, train_transforms)
    val_dataset = dataset_processing.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE,
                                                       TEST_LABEL_FILE, val_transforms)'''
    train_dataset = dataset_processing.DatasetProcessing(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE,
                                                         train_transforms)
    val_dataset = dataset_processing.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE,
                                                       val_transforms)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ## init evaluation data loader
    if evaluate_transforms is not None:
        evaluate_dataset = dataset_processing.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE,
                                                       evaluate_transforms)
        evaluate_loader = torch.utils.data.DataLoader(
            evaluate_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if evaluate_transforms is not None:
            validate(evaluate_loader, model, criterion)
        validate(val_loader, model, criterion)
        return
    # make directory for storing checkpoint files
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, LR.lr_factor, epoch)
        # train for one epoch
        trainObj1, trainObj2, top1_1, top1_2 = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        valObj1, valObj2, prec1_1, prec1_2 = validate(val_loader, model, criterion)
        # update stats
        stats_._update(trainObj1, trainObj2, top1_1, top1_2, valObj1, valObj2, prec1_1, prec1_2)
        # remember best prec@1 and save checkpoint
        is_best = prec1_1 > best_prec1
        best_prec1 = max(prec1_1, best_prec1)
        filename = []
        if args.store_model_everyepoch:
            filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
        else:
            filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        plot_curve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
    if evaluate_transforms is not None:
        model_file = os.path.join(args.modeldir, 'model_best.pth.tar')
        print("=> loading best model '{}'".format(model_file))
        print("=> start evaluation")
        best_model = torch.load(model_file)
        model.load_state_dict(best_model['state_dict'])
        validate(evaluate_loader, model, criterion)




def train(train_loader, model, criterion, optimizer, epoch):
    epoch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    loss1 = AverageMeter()
    loss2 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time !!input dim=nchw
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        #print("tensor shape:"+str(input.size()))
        # compute output
        for j in range(args.expand_num):
            np_input = input.numpy().transpose(0,2,3,1)
            #print("numpy shape:"+str(np_input.shape))
            trans_input = aug.augment_images(np_input)
            tensor_trans_input = torch.from_numpy(trans_input.transpose(0,3,1,2))
            output1, output2 = model(tensor_trans_input)
            loss1 = criterion(output1, target[:,0])
            loss2 = criterion(output2, target[:,1])
            losses1.update(loss1.item(), input.size(0))
            losses2.update(loss2.item(), input.size(0))
            prec1_1, prec1_5 = accuracy(output1, target[:,0], topk=(1, 5))
            prec2_1, prec2_5 = accuracy(output2, target[:,1], topk=(1, 5))
            # measure accuracy and record loss
            top1_1.update(prec1_1[0], input.size(0))
            top1_2.update(prec2_1[0], input.size(0))
            top5.update(prec1_5[0], input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            #loss1.backward(retain_graph=True)
            #loss2.backward()
            loss = loss1 + loss2
            loss.backward()
            #optimizer.step()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                'Prec1@1 {top1_1.val:.3f} ({top1_1.avg:.3f})\t'
                'Prec2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss1=losses1, loss2=losses2, top1_1=top1_1, 
                    top1_2=top1_2))

        # measure elapsed time
    epoch_time.update(time.time() - end)
    end = time.time()

    print('Epoch: [{0}]\t'
        'Time {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
        'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
        'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
        'Prec1@1 {top1_1.val:.3f} ({top1_1.avg:.3f})\t'
        'Prec2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})'.format(
            epoch, epoch_time=epoch_time,
            data_time=data_time, loss1=losses1, loss2=losses2, top1_1=top1_1, top1_2=top1_2, top5=top5))
    return losses1.avg, losses2.avg, top1_1.avg, top1_2.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            ## modified by jiangtao xie
            if len(input.size()) > 4:# 5-D tensor
                bs, crops, ch, h, w = input.size()
                output1, output2 = model(input.view(-1, ch, h, w))
                # fuse scores among all crops
                output1 = output1.view(bs, crops, -1).mean(dim=1)
                output2 = output2.view(bs, crops, -1).mean(dim=1)
            else:
                output1, output2 = model(input)
            loss1 = criterion(output1, target[:,0])
            loss2 = criterion(output2, target[:,1])

            # measure accuracy and record loss
            prec1_1, prec1_5 = accuracy(output1, target[:,0], topk=(1, 5))
            prec2_1, prec2_5 = accuracy(output2, target[:,1], topk=(1, 5))
            losses1.update(loss1.item(), input.size(0))
            losses2.update(loss2.item(), input.size(0))
            top1_1.update(prec1_1[0], input.size(0))
            top1_2.update(prec2_1[0], input.size(0))
            top5.update(prec1_5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Prec1@1 {top1_1.val:.3f} ({top1_1.avg:.3f})\t'
                      'Prec2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss1=losses1, loss2=losses2, 
                       top1_1=top1_1, top1_2=top1_2, top5=top5))

        print(' * Prec1@1 {top1_1.avg:.3f} Prec2@1 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))

    return losses1.avg, losses2.avg, top1_1.avg, top1_2.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    time.sleep(10)
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


class Learning_rate_generater(object):
    """Generates a list of learning rate for each training epoch"""
    def __init__(self, method, params, total_epoch):
        if method == 'step':
            lr_factor, lr = self.step(params, total_epoch)
        elif method == 'log':
            lr_factor, lr = self.log(params, total_epoch)
        else:
            raise KeyError("=> undefined learning rate method '{}'" .format(method))
        self.lr_factor = lr_factor
        self.lr = lr
    def step(self, params, total_epoch):
        decrease_until = params[0]
        decrease_num = len(decrease_until)
        base_factor = 0.1
        lr_factor = torch.ones(total_epoch, dtype=torch.double)
        lr = [args.lr]
        for num in range(decrease_num):
            if decrease_until[num] < total_epoch:
                lr_factor[int(decrease_until[num])] = base_factor
        for epoch in range(1,total_epoch):
            lr.append(lr[-1]*lr_factor[epoch])
        return lr_factor, lr
    def log(self, params, total_epoch):
        params = params[0]
        left_range = params[0]
        right_range = params[1]
        np_lr = np.logspace(left_range, right_range, total_epoch)
        lr_factor = [1]
        lr = [np_lr[0]]
        for epoch in range(1, total_epoch):
            lr.append(np_lr[epoch])
            lr_factor.append(np_lr[epoch]/np_lr[epoch-1])
        if lr[0] != args.lr:
            args.lr = lr[0]
        return lr_factor, lr


def adjust_learning_rate(optimizer, lr_factor, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    groups = ['features']
    groups.append('representation')
    groups.append('classifier')
    num_group = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_factor[epoch]
        print('the learning rate is set to {0:.5f} in {1:} part'.format(param_group['lr'], groups[num_group]))
        num_group += 1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
