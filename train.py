import torch
from torch.autograd import Variable
import time
import os
import sys
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, model_name, criterion, optimizer, opt,
                epoch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()
    training_process = tqdm(data_loader)
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

    epoch_logger.log(phase="train",values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    epoch_time = time.time() - start_time
    # print("training: epoch:{0}\t seg_acc:{1:.4f} \t using:{2:.3f} minutes".format(epoch, accuracies.avg, epoch_time / 60))
    if epoch % opt["checkpoint"] == 0:
        save_dir = os.path.join(opt["result_path"], model_name.split("/")[-1].split(".h5")[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file_path = os.path.join(save_dir,'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
#            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
            
