import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like
import pandas as pds
from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset
import pickle
import torch
import tensorboardX

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

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


class Logger(object):

    def __init__(self, model_name,header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/"+model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']
        
        for col in self.header[1:]:
            self.writer.add_scalar(phase+"/"+col,float(values[col]),int(epoch))


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


#def calculate_accuracy(outputs, targets):
#    batch_size = targets.size(0)
#
#    _, pred = outputs.topk(1, 1, True)
#    pred = pred.t()
#    correct = pred.eq(targets.view(1, -1))
#    n_correct_elems = correct.float().sum().data[0]
#
#    return n_correct_elems / batch_size

def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)

def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):
    batch_size = targets.size(0)
    y_pred = outputs[:,0,:,:,:]
    y_truth = targets[:,0,:,:,:]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps/2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union 
    
    return dice / batch_size

def load_old_model(model, optimizer, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return model, epoch, optimizer 

def normalize_data(data, mean, std):
    pass

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        tmp_img = read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation)
        image_list.append(tmp_img)
    
    return image_list

def record_padding_mat(image_files, padding_mat):
    infile_array = image_files[0].split("/")[:-1]
    new_path = "/".join(a for a in infile_array)
    out_name = os.path.join(new_path,"padding_mat.csv")
    out_file = pds.DataFrame(data=padding_mat[0])
    out_file.to_csv(out_name,header=None)
    
def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    # print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        image = resize(image, new_shape=image_shape, interpolation=interpolation)
    return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)

def get_multi_class_labels(data, n_labels, labels=None, label_containing=False):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            if not label_containing:
                y[:, label_index][data[:, 0] == labels[label_index]] = 1
            else:
                if labels[label_index] == 2:
                    y[:, label_index][data[:, 0] >= 1] = 1
                elif labels[label_index] == 1:
                    y[:, label_index][data[:, 0] == 1] = 1
                    y[:, label_index][data[:, 0] == 4] = 1
                elif labels[label_index] == 4:
                    y[:, label_index][data[:, 0] == 4] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y