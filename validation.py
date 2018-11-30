'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''

import torch
from torch.autograd import Variable
import time
import sys
from tqdm import tqdm
from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_set, model, criterion, optimizer, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    data_set.file_open()
    valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                     batch_size=opt["validation_batch_size"], 
                                     shuffle=False,
                                     pin_memory=True)
    val_process = tqdm(valildation_loader)
    start_time = time.time()
    for i, (inputs, targets) in enumerate(val_process):
        if i > 0:
            val_process.set_description("Loss: %.4f, Acc: %.4f"%(losses.avg, accuracies.avg))
        if opt["cuda_devices"] is not None:
            #targets = targets.cuda(async=True)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        with torch.no_grad():
            if opt["VAE_enable"]:
                outputs, distr = model(inputs)
                loss = criterion(outputs, targets, distr)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    epoch_time = time.time() - start_time
    data_set.file_open()
    print("validation: epoch:{0}\t seg_acc:{1:.4f} \t using:{2:.3f} minutes".format(epoch, accuracies.avg, epoch_time / 60))
    
    logger.log(phase="val",values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg, accuracies.avg