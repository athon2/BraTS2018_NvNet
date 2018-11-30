'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''
import torch
from torch.autograd import Variable
import time
import os
import sys
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    
    
    data_set.file_open()
    train_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                               batch_size=opt["batch_size"], 
                                               shuffle=True, 
                                               pin_memory=True)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Loss: %.4f, Acc: %.4f"%(losses.avg.item(), accuracies.avg.item()))

        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        if opt["VAE_enable"]:
            outputs, distr = model(inputs)
            loss = criterion(outputs, targets, distr)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs.cpu(), targets.cpu())
        losses.update(loss.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.log(phase="train",values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    data_set.file_close()
            
