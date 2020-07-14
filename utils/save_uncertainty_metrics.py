import argparse
import os
import numpy as np
import csv

from PIL import Image
from scipy.special import logsumexp

def csv_writer(metric_dict, metric_name, save_path):
    '''Saves the uncertainty metrics for each image in csv format in disks.
    Args:
    metric_dict: uncertainty metric as dictionary containing filename and value
    metric_name: uncertainty metric name
    save_path: Save path in the disk
    '''
    with open(os.path.join(save_path, metric_name+'.csv'), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in metric_dict.items():
            writer.writerow([key, value])

def mi_metric(prd):
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
    #parser.add_argument('--samples_size', type=int, default=50)
    args = parser.parse_args()
    
    if args.sampling_strategy == 'hse':
        path_samples = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/hse/'
    if args.sampling_strategy == 'dropout':
        path_samples= '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/dropout/'
    if args.sampling_strategy == 'augment':
        path_samples = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/augment/'

    
    path_samples_sorted = (sorted(os.listdir(path_samples),key=sort_by_name)) 
    TEST_FN_PATH = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt' 
    test_fn = open(TEST_FN_PATH, 'r')
    test_fn_list = [line.split() for line in test_fn.readlines()]
    test_fn.close()
    print (len(test_fn_list))
    
    save_path =  '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/metrics/csv_files/'
    if not (os.path.isdir(save_path)) :
         os.mkdir(save_path)

    samples_size = [10,20,30,40,50]
    for samples in samples_size:
        current_sample_path = path_samples_sorted[0:samples]
        print ('Total No of samples are {}'.format(len(current_sample_path)))
        print ('*'  * 60)
        entropy_dict = {} ; variance_dict = {} ; mi_dict = {} 
        for count in range(len(test_fn_list)):
            img_path = os.path.join(path_samples, 'sample_' + str(0) , test_fn_list[count][0] + '.png' )
            img = Image.open(img_path)
            width, height = img.size
            all_imgs = np.zeros((len(current_sample_path),height,width))
            image_count = 0 
            for samples_folders in current_sample_path:
                img_path = os.path.join(path_samples, samples_folders , test_fn_list[count][0] + '.png' )
                print (img_path)
                img = Image.open(img_path)
                img_array = np.array(img)
                img_array = img_array /255.0
                all_imgs[image_count,:,:] = img_array
                image_count = image_count+1
            entropy_val = entropy(all_imgs)
            entropy_log =logsumexp(entropy_val) / (img.size[0]* img.size[1])
            print ('the calculated log sum of entropy is', entropy_log)
            variance_val = variance(all_imgs)
            variance_log =logsumexp(variance_val) / (img.size[0]* img.size[1])
            print ('the calculated log sum of variance is', variance_log)
            mi_val = mi_metric(all_imgs)
            mi_log =logsumexp(mi_val) / (img.size[0]* img.size[1])
            print ('the calculated log sum of mi is', mi_log)
            print ('***************************************************************')
            entropy_dict[test_fn_list[count][0]] =entropy_log
            variance_dict[test_fn_list[count][0]] =variance_log
            mi_dict[test_fn_list[count][0]] =mi_log
        

        save_entropy_csv = csv_writer(entropy_dict, 'entropy_'  + args.sampling_strategy + '_' + 'samples_' + str(samples), save_path)
        save_variance_csv = csv_writer(variance_dict, 'variance_'+ args.sampling_strategy+ '_' + 'samples_' + str(samples), save_path)
        save_mi_csv = csv_writer(mi_dict, 'mi_'+ args.sampling_strategy + '_' + 'samples_' + str(samples), save_path)
    
    




