import os
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from PIL import Image

def mi_uncertainty(prd):
    # https://arxiv.org/pdf/1703.02910.pdf
    mcs = np.repeat(np.expand_dims(prd, -1), 2, -1)
    mcs[..., 0] = 1 - mcs[..., 1]  # 10, nb_img, 192, 192, 64, 2

    entropy = -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)
    expected_entropy = -np.mean(np.sum(mcs * np.log(mcs + 1e-5), -1), 0)
    mi = entropy - expected_entropy
    return mi


def entropy(prd_mcs):
    mcs = np.repeat(np.expand_dims(prd_mcs, -1), 2, -1)
    mcs[..., 0] = 1 - mcs[..., 1]  # 10, 50, 192, 192, 64, 2
    return -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)


def variance(array):
    sd_pixel = np.std(array, axis =0)
    return sd_pixel


def prob_atlas(array):
    atlas = np.mean(array, axis =0)
    return atlas

def sort_by_name(file):
    temp = file.split("_")
    print(temp[1])
    return int(temp[1])

def save_heat_map(img, save_path, uncertainty_metric):
    plt.imshow(img, cmap = 'magma')
    cbar=plt.colorbar()
    cbar.set_label('uncertainty - ' + uncertainty_metric)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=500, pad_inches =0)
    plt.clf()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Saving Uncertainty Images and Color Map")

    parser.add_argument('--is_camus', action='store_true', default=False)
    parser.add_argument('--is_dynamic', action='store_true', default=False)

    parser.add_argument('--is_dropout', action='store_true', default=False)
    parser.add_argument('--is_epoch', action='store_true', default=False)
    parser.add_argument('--is_augment', action='store_true', default=False)

    parser.add_argument('--is_variance', action='store_true', default=False)
    parser.add_argument('--is_entropy', action='store_true', default=False)
    parser.add_argument('--is_mi', action='store_true', default=False)
    parser.add_argument('--is_atlas', action='store_true', default=False)

    args = parser.parse_args()
    if args.is_camus:
        if args.is_epoch:
            images_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/all_epochs_softmax_pred/val_set/'
            files_path = images_path + 'epoch_300/'
            fnames = os.listdir(files_path)
            save_base_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/val_set/softmax_epochs/'
        if args.is_dropout:
            images_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/all_dropout_softmax_pred/val_set/'  
            files_path = images_path + 'dropout_0/'
            fnames = os.listdir(files_path)
            save_base_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/val_set/softmax_dropout/' 
        if args.is_augment: 
            images_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/all_augment_softmax_pred/val_set/'   
            files_path = images_path + 'augment_0/' 
            fnames = os.listdir(files_path)
            save_base_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/val_set/softmax_augment/'

    if args.is_dynamic:
        if args.is_epoch:         
            images_path = '/media/HDD1/lavsen/ouyang/results/all_epochs_softmax_pred/' #Dropout
            files_path = images_path + 'epoch_3/'
            fnames = os.listdir(files_path)
            #save_base_path = '/media/HDD1/lavsen/results/dynamic/uncertainty_map/softmax_epochs_100/test_set/deeplab_dynamic_no_dropout/' 
            save_base_path = '/media/HDD1/lavsen/ouyang/results/uncertainty_images/softmax_epochs/test_set/'
        if args.is_dropout:
            images_path = '/media/HDD1/lavsen/results/camus/deeplab_dynamic_no_dropout/all_dropout_softmax_pred/test_set/'
            files_path = images_path + 'dropout_0/'
            fnames = os.listdir(files_path)
            save_base_path = '/media/HDD1/lavsen/results/camus/deeplab_dynamic_no_dropout/uncertainty_images/dropout/test_set/variance/' 
        if args.is_augment: 
            images_path = '/media/HDD1/lavsen/ouyang/results/all_augment_sigmoid_pred/' 
            files_path = images_path + 'augment_0/'
            fnames = os.listdir(files_path)
            save_base_path = '/media/HDD1/lavsen/ouyang/results/uncertainty_images/softmax_augment/test_set/'
       

    dir_path = sorted(os.listdir(images_path),key=sort_by_name)


    print (dir_path)
    #dir_path = dir_path[0:22]
   

    if args.is_variance:
        save_path = save_base_path + 'variance/'
        save_path_cm = save_base_path + 'variance_cm/'
        uncertainty_metric = 'variance'
    if args.is_entropy:
         save_path = save_base_path + 'entropy/'
         save_path_cm = save_base_path + 'entropy_cm/'
         uncertainty_metric = 'entropy'
    if args.is_mi:
        save_path = save_base_path + 'mi/'
        save_path_cm = save_base_path +'mi_cm/'
        uncertainty_metric = 'mi'
    
    if args.is_atlas:
        save_path = save_base_path + 'atlas/'
        save_path_cm = save_base_path +'atlas_cm/'
        uncertainty_metric = 'atlas'

   
    for count in range(len(fnames)):
        img_ref_path = files_path + fnames[count]
        img_ref = Image.open(img_ref_path)
        width, height = img_ref.size
        all_imgs = np.zeros((len(dir_path),height,width))
        i_count=0
        for folder in dir_path:
            fname = fnames[count]
            file_path = images_path + folder + '/' + fname
            print (file_path)
            img = Image.open(file_path)
            img_array = np.array(img)
            img_array = img_array /255.0
            all_imgs[i_count,:,:] = img_array
            i_count = i_count+1
        if args.is_variance:
            result = variance(all_imgs)
        if args.is_entropy:
            result = entropy(all_imgs)
        if args.is_mi:
            result = mi_uncertainty(all_imgs)
        if args.is_atlas:
            result = prob_atlas(all_imgs)
        print (result.shape)
        save_heat_map(result, save_path_cm + fname, uncertainty_metric)

        result = (result *255).astype(np.uint8)
        img_sd = Image.fromarray(result)
        img_sd.save(save_path + fname)
        del all_imgs
