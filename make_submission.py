import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import glob
from nilearn.image import new_img_like
from utils.utils import read_image,resize,get_multi_class_labels
from utils.normalize import find_downsized_info
from predict import config
def get_cases(cases_dir):
    cases_list = glob.glob(os.path.join(cases_dir, "*", "postprocessed_prediction.nii.gz"))
    return cases_list

def cal_crop_shape(crop_slices):
    crop_shape = list()
    for dim in crop_slices:
        crop_shape.append(dim.stop - dim.start)
    return tuple(crop_shape)

def combine_labels(data, labels=(1,2,4), threshold=500):
    label_indices = (1,0,2)
    combined_data = np.zeros_like(data[0])
    assert combined_data.shape == (128,128,128), "Wrong dimession!"
    for idx in label_indices:
        cur_label = labels[idx]
        count = np.sum(data[idx] == 1)
        if count < threshold and idx == 2:
            continue
        combined_data[data[idx] == 1] = cur_label 
        
    return combined_data

def postprocessing(case_dir):
    print("postprocessing ...")
    cases_list = glob.glob(os.path.join(cases_dir, "*", "prediction.nii.gz"))
    
    for case in tqdm(cases_list):
        net_pred_image = nib.load(case)
        net_pred_data = net_pred_image.get_data()
        # Get multi-class labels data
        multi_label_data = np.squeeze(get_multi_class_labels(net_pred_data[np.newaxis][np.newaxis], n_labels=3, labels=(1,2,4), label_containing=True))
        assert multi_label_data.shape == (3, 128, 128, 128),  "Wrong shape!Excepeted (3, 128, 128, 128) but got{0}.".format(multi_label_data.shape)
        # Fill holes iteratively utill no voxel value changes
        iter_data = multi_label_data.copy()
        iteration = 1
        while True:
            tmp_data = np.zeros_like(iter_data)
            for label_channel in range(iter_data.shape[0]):    
                if iteration > 2 and label_channel == 2:
                    tmp_data[label_channel] = iter_data[label_channel]
                else:
                    tmp_data[label_channel] = ndimage.binary_fill_holes(iter_data[label_channel]).astype(np.int8)
            if np.sum(np.logical_xor(tmp_data,iter_data)) == 0:
                filled_data = iter_data.copy()
                break
            else:
                iter_data = tmp_data.copy()
                iteration += 1
        
        combined_data = combine_labels(filled_data)
        assert combined_data.shape == (128, 128, 128), "Wrong shape!Excepeted (128, 128, 128) but got{0}.".format(combined_data.shape)
        new_pred_image = new_img_like(net_pred_image, combined_data)
        case_id = case.split("/")[-2]
        new_pred_image.to_filename(os.path.join(case_dir, case_id,"postprocessed_prediction.nii.gz"))
        
    
    
def reconstruct(cases_list, data_src, output_dir, crop_size=(128, 128, 128), original_shape=(240,240,155)):
    print("reconstructing...")
    for case in  tqdm(cases_list):
        case_id = case.split("/")[-2]
        data_path = os.path.join(data_src, case_id, "*")
        set_files = glob.glob(data_path)
        set_files.remove(os.path.join(data_src, case_id, "truth.nii.gz"))
        crop_slices, affine, header = find_downsized_info([set_files], crop_size)
        crop_shape = cal_crop_shape(crop_slices)
        pred_image = read_image(case)
        pred_image_data = pred_image.get_data()
        pred_image_data = np.flip(pred_image_data, axis=0)
        pred_image_data = np.flip(pred_image_data, axis=1)
        fixed_pred_image = new_img_like(pred_image, pred_image_data,affine=pred_image.affine)
        
        resized_image = resize(fixed_pred_image, crop_shape)
        fill_data = np.zeros(original_shape)
        fill_data[crop_slices[0], crop_slices[1], crop_slices[2]] = resized_image.get_data()
        origin_image = read_image(set_files[0])
        fill_image = new_img_like(origin_image, fill_data, affine=origin_image.affine)
        fill_image.to_filename(os.path.join(output_dir, case_id+".nii.gz"))
        # break

if __name__ == "__main__":
    data_src = "./data/BraTs_2018_Data_Validation/"    # original .nii.gz files
    output_dir = os.path.join(config["prediction_dir"],"Reconstruct") # reconstructed results (128,128,128) => (240, 240, 155)
    cases_dir = os.path.join(config["prediction_dir"],config["model_file"].split(".h5")[0]) # prediction results (128,128,128)

    postprocessing(cases_dir)
    cases_list = get_cases(cases_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reconstruct(cases_list, data_src, output_dir)