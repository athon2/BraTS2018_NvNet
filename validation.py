import torch
from torch.autograd import Variable
import time
import sys
from tqdm import tqdm
from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()
    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        data_time.update(time.time() - end_time)

        if opt["cuda_devices"] is not None:
            #targets = targets.cuda(async=True)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        with torch.no_grad():
            outputs, distr = model(inputs)
        loss = criterion(outputs.cpu(), targets.cpu(), distr.cpu())
        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    epoch_time = time.time() - start_time
    print("validation: epoch:{0}\t seg_acc:{1:.4f} \t using:{2:.3f} minutes".format(epoch, accuracies.avg, epoch_time / 60))
    logger.log(phase="val",values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg