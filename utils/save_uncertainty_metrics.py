import argparse
import os
import numpy as np
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
    var = np.var(array, axis =0)
    return var

def sort_by_name(file):
    temp = file.split("_")   #All foldernames are split by _
    return int(temp[1])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Computing and saving uncertainty metrics")
    parser.add_argument('--sampling_strategy', type=str, default=None, choices=['hse', 'dropout', 'augment'])
    parser.add_argument('--no_samples', type=int, default=50)
    args = parser.parse_args()
    
    if args.sampling_strategy == 'hse':
        path_samples = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/hse/'
    if args.sampling_strategy == 'dropout':
        path_samples= '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/dropout/'
    if args.sampling_strategy == 'augment':
        path_samples = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/augment/'

    samples_path = (sorted(os.listdir(path_samples),key=sort_by_name))[0:args.no_samples]
    print ('Total No of samples are {}'.format(len(samples_path)))

    TEST_FN_PATH = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt' 
    test_fn = open(TEST_FN_PATH, 'r')
    test_fn_list = [line.split() for line in test_fn.readlines()]
    test_fn.close()
    print (len(test_fn_list))
    for count in range(len(test_fn_list)):
        img_path = path_samples + 'sample_' + str(count) + '/'  + test_fn_list[count][0] + '.png'
        img = Image.open(img_path)
        width, height = img.size
        all_imgs = np.zeros((len(samples_path),height,width))
        i_count=0
        for samples in samples_path:
             sample_path = path_samples + 'sample_' + str(count) + '/'  + test_fn_list[count][0] + '.png'
             print (sample_path)

       
    # TO DO Complete code



