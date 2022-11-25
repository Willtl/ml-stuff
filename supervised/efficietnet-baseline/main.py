import os
import sys
import time

import torch

import utils
from options import parse_option
from utils import AverageMeter


# training iteration
def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)


        # feed forward and compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), images.shape[0])
        acc1, acc5 = utils.accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0].item(), images.shape[0])

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, top1.avg


def test(args, test_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), images.shape[0])
            acc1, acc5 = utils.accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0].item(), images.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main(args):
    # Split image dataset into folders
    utils.split_image_dataset()

    # Load dataset, and create dataloaders for training
    # train_loader, test_loader = utils.set_loader(args, augment_strategy='custom', plot_augments=False)
    train_loader, test_loader = utils.load_data(args)

    # Load model and prepare for training
    model, criterion = utils.set_model(args, train_all_params=True)

    # Set optimization related stuff
    optimizer = utils.set_optimizer(args, model)
    main_lr_scheduler = utils.get_scheduler(args, optimizer)
    lr_scheduler = utils.get_warmup_scheduler(args, main_lr_scheduler, optimizer)
    print('lr_scheduler', lr_scheduler)

    # Training loop
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        t1 = time.time()
        train_loss, train_acc = train_one_epoch(args, train_loader, model, criterion, optimizer, epoch)
        t2 = time.time()
        lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Stats. epoch: {epoch}: Loss: {train_loss:.4f}, '
              f'Train Acc@1: {train_acc:.4f}, lr: {cur_lr:.4f}, Time: {t2 - t1:.4f}')

        # evaluation
        test_loss, test_acc = test(args, test_loader, model, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
        print(f'Best Acc@1: {best_acc:.4f}')

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            utils.save_model(args, model, optimizer, epoch, save_file)

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pth')
    utils.save_model(args, model, optimizer, args.epochs, save_file)


if __name__ == '__main__':
    args = parse_option()
    main(args)
