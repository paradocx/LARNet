############################  general packages
import argparse
import os,sys,shutil
import time
############################ NN  packages
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
############################  resnet 50
from ResNet import resnet50
from Otheroperation import MS1MV2, CaffeCrop


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='End2end parameters')
parser.add_argument('--data_dir', default='../../data/end2end/MS1MV2', type=str, help='path of the training dataset')
parser.add_argument('--type',  default='resnet50', choices=model_names, help='resnet arch: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--workers', default=8, type=int, help='data loading workers-8')
parser.add_argument('--epochs', default=100, type=int, help='total epochs')
parser.add_argument('--start_epoch', default=0, type=int, help='for restart')
parser.add_argument('--bs', '--batch_size', default=256, type=int, help='batch size-256')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='learning rate-0.1')
parser.add_argument('--mot', '--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wtd', '--weight_decay', default=1e-4, type=float,  help='weight-1e-4)')


parser.add_argument('--pretrained', default='./', type=str, help='path to checkpoint')
parser.add_argument('--retrain', default='./', type=str, help='restart train')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--model_dir', default='./model', type=str, help='path to store model')
parser.add_argument('--end2end', action='store_true', help='if end2end')

best_prec1 = 0



if __name__ == '__main__':
    global args, best_prec1
    args = parser.parse_args()

    print('data_dir:', args.data_dir)
    print('if end2end? :', args.end2end)

    # load dataset
    train_list_file = args.data_dir+'train_list.txt'               #####the path of img list(training)
    train_label_file = args.data_dir+'train_label.txt'             #####the path of label list(training)
    caffe_crop = CaffeCrop('train')                                #####resize and crop



    train_dataset =  MS1MV2(args.img_dir, train_list_file, train_label_file, transforms.Compose([caffe_crop,transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True)
   
    caffe_crop = CaffeCrop('test')
    test_list_file = args.data_dir+'test_list.txt'                #####the path of img list(teset)
    test_label_file = args.data_dir+'test_label.txt'              #####the path of label list(teset)
    test =  MS1MV2(args.img_dir, test_list_file, test_label_file, transforms.Compose([caffe_crop,transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)

    assert(train_dataset.max_label == test_dataset.max_label)
    class_num = train_dataset.max_label + 1

    print('class_num: ',class_num)
    
    
    # model
    model = None
    assert(args.arch in ['resnet50'])         #####maybe other model
    if args.arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, end2end=args.end2end)
    model = torch.nn.DataParallel(model).cuda()
    

    # loss function 
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.mot,
                                weight_decay=args.wtd)

   # optionally checkpoint load or restart
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        
        for key in pretrained_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict)


    if args.retrain:
        if os.path.isfile(args.retrain):
            print("=> loading checkpoint '{}'".format(args.retrain))
            checkpoint = torch.load(args.retrain)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.retrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.retrain))

    cudnn.benchmark = True

   
#######################################################################test
    if args.evaluate:
        validate(test_loader, model, criterion)
        return

#######################################################################train
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)                    #####The strategy of lr

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec1 = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)





def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    angle_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, angle) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        angle = angle.float().cuda(async=True)
        input_var = torch.autograd.Variable(input)
        angle_var = torch.autograd.Variable(angle)
        target_var = torch.autograd.Variable(target)

        # compute output
        pred_score = model(input_var, angle_var)

        loss = criterion(pred_score, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_score.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(test_loader, model, criterion):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    angle_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, angle) in enumerate(test_loader):
        target = target.cuda(async=True)
        angle = angle.float().cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        angle_var = torch.autograd.Variable(angle)

        # compute output
        pred_score = model(input_var, angle_var)

        loss = criterion(pred_score, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_score.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses, 
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%5==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.8), int(args.epochs*0.9), int(args.epochs*0.95)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



