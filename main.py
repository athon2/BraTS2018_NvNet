# main.py
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from utils import Logger
from train import train_epoch
from validation import val_epoch
from nvnet import NvNet
from metrics import CombinedLoss
from dataset import BratsDataset
config = dict()
config["cuda_devices"] = 1
config["labels"] = (2,)
config["model_file"] = os.path.abspath("single_label_2_augment.h5")
config["initial_learning_rate"] = 5e-4
config["batch_size"] = 1
config["image_shape"] = (128, 128, 128)
# config["labels"] = (1, 2, 4)
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["loss_k1_weight"] = 0.1
config["loss_k2_weight"] = 0.1
config["random_offset"] = True
config["random_flip"] = True  # augments the data by randomly flipping an axis during generating a data
# config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["result_path"] = "./checkpoint_models/"
config["data_file"] = os.path.abspath("isensee_mixed_brats_data.h5")
config["training_file"] = os.path.abspath("isensee_mixed_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config["L2_norm"] = 1e-5
config["patience"] = 5
config["epochs"] = 100
config["checkpoint"] = 3
config["label_containing"] = True
def load_old_model(model_path):
    pass


def main():
    # init or load model
    print("init model")
    print("input shape:",config["input_shape"])
    if not config["overwrite"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = NvNet(input_shape=config["input_shape"], seg_outChans=config["n_labels"])
    
    parameters = model.parameters()
    loss_function = CombinedLoss(k1=config["loss_k1_weight"], k2=config["loss_k2_weight"])
    # data_generator
    print("data generating")
    training_data = BratsDataset(phase="train", config=config)
    train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                               batch_size=config["batch_size"], 
                                               shuffle=True, 
                                               pin_memory=True)
    valildation_data = BratsDataset(phase="validate", config=config)
    valildation_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                               batch_size=config["batch_size"], 
                                               shuffle=False, 
                                               pin_memory=True)
    
    train_logger = Logger(os.path.join("./logs/", config["model_file"].split(".h5")[0]+"_train.log"), 
                          ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(os.path.join("./logs/", config["model_file"].split(".h5")[0]+"_batch.log"), 
                          ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    
    valdate_logger = Logger(os.path.join("./logs/", config["model_file"].split(".h5")[0]+"_val.log"), 
                          ['epoch', 'loss', 'acc'])
    
    if config["cuda_devices"] is not None:
        model = model.cuda(config["cuda_devices"])
        loss_function = loss_function.cuda(config["cuda_devices"])
        
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay = config["L2_norm"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config["patience"])
    print("training on label:{}".format(config["labels"]))    
    for i in range(config["epochs"]):
        train_epoch(epoch=i, 
                    data_loader=train_loader, 
                    model=model,
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    epoch_logger=train_logger, 
                    batch_logger=train_batch_logger)
    
        val_epoch(epoch=i, 
                  data_loader=valildation_loader, 
                  model=model, 
                  criterion=loss_function, 
                  opt=config, 
                  logger=valdate_logger)
        
if __name__ == '__main__':
    main()
