import os
import uuid
from PIL import Image
import torchvision
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import eval_dataset
from imgaug import augmenters as iaa
from eval_dataset import *
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

def main():
    global args, best_prec1
    global path
    path = "/workspace/mnt/group/video/linshaokang/fast-MPN-COV/errorimgs"
    args = parser.parse_args()
    print(args)

    # create model
    if args.representation == 'MPNCOV':
        representation = {'function':MPNCOV,
                          'iterNum':5,
                          'is_sqrt':True,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':None}

    model = get_model(args.arch,
                      representation,
                      args.num_classes,
                      args.freezed_layer,
                      pretrained=args.pretrained)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

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

    _, _, evaluate_transforms = preprocess_strategy(args.benchmark)

    evaluate_dataset = eval_dataset.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE,
                                                       evaluate_transforms)
    evaluate_loader = torch.utils.data.DataLoader(
        evaluate_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #if args.evaluate:
    #    validate(evaluate_loader, model, criterion)

    if evaluate_transforms is not None:
        model_file = os.path.join(args.modeldir, 'model_best.pth.tar')
        print("=> loading best model '{}'".format(model_file))
        print("=> start evaluation")
        best_model = torch.load(model_file)
        model.load_state_dict(best_model['state_dict'])
        validate(evaluate_loader, model, criterion)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, filenames) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            ## modified by jiangtao xie
            output1, output2 = model(input)
            store_errorimgs(output1, output2, target, filenames)
            loss1 = criterion(output1, target[:,0])
            loss2 = criterion(output2, target[:,1])

            # measure accuracy and record loss
            prec1_1, _ = accuracy(output1, target[:,0], topk=(1, 5))
            prec2_1, _ = accuracy(output2, target[:,1], topk=(1, 5))
            losses1.update(loss1.item(), input.size(0))
            losses2.update(loss2.item(), input.size(0))
            top1_1.update(prec1_1[0], input.size(0))
            top1_2.update(prec2_1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Prec1@1 {top1_1.val:.3f} ({top1_1.avg:.3f})\t'
                      'Prec2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss1=losses1, loss2=losses2, 
                       top1_1=top1_1, top1_2=top1_2))

        print(' * Prec1@1 {top1_1.avg:.3f} Prec2@1 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))

    return losses1.avg, losses2.avg, top1_1.avg, top1_2.avg

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
        
def save_image(filename, cloth, action, path):
    unloader = torchvision.transforms.ToPILImage()
    dir = path
    pth = os.path.join(dir,str(cloth),str(action))
    if not os.path.exists(pth):
        os.makedirs(pth)
    shutil.copyfile(filename, os.path.join(pth, filename[filename.rfind('/')+1:]))
    
def store_errorimgs(output1, output2, target, filenames):
    with torch.no_grad():
        #output = torch.Tensor(8,5)
        #target = torch.ones(8,2,dtype=torch.long)
        #target = target[:,0]
        target1 = target[:,0]
        target2 = target[:,1]
        batch_size = target.size(0)
        _, pred1 = output1.topk(1, 1, True, True)
        pred1 = pred1.t()
        correct1 = pred1.eq(target1.view(1, -1).expand_as(pred1))
        correct1 = correct1.cpu().view(-1).numpy()  #classification result eg [0 1 0 1 0 0 0]
        res1 = pred1[0,:].cpu().numpy() #prediction result eg [0 3 1 2 0 1 4]
        _, pred2 = output2.topk(1, 1, True, True)
        pred2 = pred2.t()
        correct2 = pred2.eq(target2.view(1, -1).expand_as(pred2))
        correct2 = correct2.cpu().view(-1).numpy()  #classification result eg [0 1 0 1 0 0 0]
        res2 = pred2[0,:].cpu().numpy() #prediction result eg [0 3 1 2 0 1 4]
        correct = correct1 + correct2
        file=open('/workspace/mnt/group/video/linshaokang/fast-MPN-COV/correct.txt','w') 
        file.write(str(correct)); 
        file.close() 
        for idx in range(batch_size):
            if correct[idx] != 2:
                save_image(filenames[idx], res1[idx], res2[idx], path)
            #print(correct,res)
        #return correct,res

if __name__ == '__main__':
    main()
