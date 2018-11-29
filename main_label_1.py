# main.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from tqdm import tqdm
from utils import Logger,load_old_model
from train import train_epoch
from validation import val_epoch
from nvnet import NvNet
from metrics import CombinedLoss, SoftDiceLoss
from dataset import BratsDataset
config = dict()
config["cuda_devices"] = True
config["labels"] = (1,)
config["model_file"] = os.path.abspath("single_label_{}_dice.h5".format(config["labels"][0]))
config["initial_learning_rate"] = 1e-5
config["batch_size"] = 1
config["validation_batch_size"] = 1
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
# config["saved_model_file"] = os.path.abspath("./checkpoint_models/single_label_2_flip/save_55.pth")
config["saved_model_file"] = None
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config["L2_norm"] = 1e-5
config["patience"] = 2
config["lr_decay"] = 0.7
config["epochs"] = 300
config["checkpoint"] = 1
config["label_containing"] = True
config["VAE_enable"] = False


def main():
    # init or load model
    print("init model with input shape",config["input_shape"])
    model = NvNet(config=config,input_shape=config["input_shape"], seg_outChans=config["n_labels"])
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay = config["L2_norm"])
    start_epoch = 1
    if config["VAE_enable"]:
        loss_function = CombinedLoss(k1=config["loss_k1_weight"], k2=config["loss_k2_weight"])
    else:
        loss_function = SoftDiceLoss()
    # data_generator
    print("data generating")
    training_data = BratsDataset(phase="train", config=config)
    # train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                               # batch_size=config["batch_size"], 
                                               # shuffle=True, 
                                               # pin_memory=True)
    valildation_data = BratsDataset(phase="validate", config=config)
    # valildation_loader = torch.utils.data.DataLoader(dataset=valildation_data, 
                                               # batch_size=config["batch_size"], 
                                               # shuffle=True, 
                                               # pin_memory=True)
    
    train_logger = Logger(model_name=config["model_file"],header=['epoch', 'loss', 'acc', 'lr'])

    if config["cuda_devices"] is not None:
        model = model.cuda()
        loss_function = loss_function.cuda()
        
    # if not config["overwrite"] and os.path.exists(config["model_file"]) or os.path.exists(config["saved_model_file"]):
    #    model, start_epoch, optimizer = load_old_model(model, optimizer, saved_model_path=config["saved_model_file"])
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"],patience=config["patience"])
    
    print("training on label:{}".format(config["labels"]))    
    for i in range(start_epoch,config["epochs"]):
        train_epoch(epoch=i, 
                    data_set=training_data, 
                    model=model,
                    model_name=config["model_file"], 
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    epoch_logger=train_logger) 
        
        val_loss = val_epoch(epoch=i, 
                  data_set=valildation_data, 
                  model=model, 
                  criterion=loss_function, 
                  opt=config,
                  optimizer=optimizer, 
                  logger=train_logger)
        scheduler.step(val_loss)
        
if __name__ == '__main__':
    main()
