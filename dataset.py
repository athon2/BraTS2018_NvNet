import os 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pickle
from utils import pickle_load

class BratsDataset(Dataset):
    def __init__(self, phase, config):
        super(BratsDataset, self).__init__()
        
        self.config = config
        self.phase = phase
        self.data_file = open_data_file(config["data_file"])
        if phase == "train":
            self.data_ids = config["traing_file"]
        elif phase == "validate":
            self.data_ids = config["validation_file"]
        elif phase == "test":
            self.data_ids = config["test_file"]
            
        self.data_list = pickle_load(self.data_ids)
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass