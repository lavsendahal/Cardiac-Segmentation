import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


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


def save_heat_map(img, save_path, uncertainty_metric):
    plt.imshow(img, cmap = 'magma')
    cbar=plt.colorbar()
    cbar.set_label('uncertainty - ' + uncertainty_metric)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=500, pad_inches =0)
    plt.clf()

if __name__ == '__main__':

    is_variance = 0
    is_entropy = 0
    is_mi =1
    is_atlas = 0
    images_path = '/media/HDD1/lavsen/results/all_epochs_softmax_pred/deeplab_lv_both_all_3_struct/test_set/'
    dir_path = sorted(os.listdir(images_path))
    dir_path = dir_path[0:100]
    print ('No of folders for uncertainty predictions', len(dir_path))
    print (dir_path)

    files_path = '/media/HDD1/lavsen/results/all_epochs_softmax_pred/deeplab_lv_both_all_3_struct/test_set/epoch_200/'
    fnames = os.listdir(files_path)
    print ('Total files', len(fnames))

    base_path = '/media/HDD1/lavsen/results/uncertainty_map/softmax_epochs_100/train_3_structures_dropout/test_set/' 
    
    if is_variance:
        save_path = base_path + 'variance/'
        save_path_cm = base_path + 'variance_cm/'
        uncertainty_metric = 'variance'
    if is_entropy:
         save_path = base_path + 'entropy/'
         save_path_cm = base_path + 'entropy_cm/'
         uncertainty_metric = 'entropy'
    if is_mi:
        save_path = base_path + 'mi/'
        save_path_cm = base_path +'mi_cm/'
        uncertainty_metric = 'mi'
    
    if is_atlas:
        save_path = base_path + 'atlas/'
        save_path_cm = base_path +'atlas_cm/'
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
        if is_variance:
            result = variance(all_imgs)
        if is_entropy:
            result = entropy(all_imgs)
        if is_mi:
            result = mi_uncertainty(all_imgs)
        if is_atlas:
            result = prob_atlas(all_imgs)
        print (result.shape)
        save_heat_map(result, save_path_cm + fname, uncertainty_metric)

        result = (result *255).astype(np.uint8)
        img_sd = Image.fromarray(result)
        img_sd.save(save_path + fname)
        del all_imgs
