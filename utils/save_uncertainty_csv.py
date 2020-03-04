import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.measure import regionprops
from scipy.special import logsumexp
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def normalize_list(given_list):
    '''
    Given a list, returns the max-min normalized list.
    '''
    amin, amax = min(given_list), max(given_list)
    for i, val in enumerate(given_list):
        given_list[i] = (val-amin) / (amax-amin)
    return given_list



def get_data_for_box_plot(df_uncertainty, df_gt_and_pred, save_path, interval, uncertainty_type):
    '''
    atlas_and_pred_dice - A pandas dataframe for atlas and pred -  dice
    gt_and_pred_dice - A pandas dataframe for gt and pred - dice
    save_path - Path where the box plots gets saved
    interval - integer (int) for different intervals of samples to plot box plot.
    Saves the box plot for different no of samples intervals.
    Returns the pandas dataframe containing dice of different intervals given by list x and corresponding dice coefficient.
    '''
    uncertainty_type='Dice_Coefficient'
    sorted_df_uncertainty = df_uncertainty.sort_values(by=uncertainty_type, ascending=False)
    sorted_df_uncertainty_file_list = sorted_df_uncertainty['all_names'].tolist()
    all_dice = []
    y =[]
    x=[]

    for i in range(interval,55,interval):
        current_list = sorted_df_uncertainty_file_list[0:i]
        dsc=[]
        for filenames in current_list:
            array = [filenames]
            dice_coef =  (df_gt_and_pred.loc[df_gt_and_pred['all_names'].isin(array)])['Dice_Coefficient'].values
            dsc.append(dice_coef[0])
        all_dice.append(dsc)
        print ('Average Dice', sum(dsc)/len(dsc))
        y.append(sum(dsc)/len(dsc))
        x.append(str(i))

    return pd.DataFrame(all_dice), x, y


def save_box_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method):

    df, x, _ = get_data_for_box_plot(df_uncertainty, df_gt_and_pred, save_path, interval, method)
    df1_transposed = df.transpose()
    df1_transposed.columns =x
    _=df1_transposed.boxplot()
    plt.xlabel('No of Samples')
    plt.ylabel('DSC')
    plt.show()
    plt.savefig(save_path + save_type)

def save_line_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method):
    _, x, y = get_data_for_box_plot(df_uncertainty, df_gt_and_pred, save_path, interval, method)
   
    x = [int(i) for i in x]
    plt.xlabel('Retained Samples')
    plt.ylabel('DSC')
    plt.plot(x, y)
    plt.savefig(save_path + save_type)

def get_uncertainty_of_image(image_path, pred_path):
    img = np.array(Image.open(image_path)).astype(np.float32)
    temp_img = np.zeros((img.shape[0], img.shape[1]))

    img_pred = np.array(Image.open(pred_path)).astype(int)
    props_img = regionprops(img_pred)
    bbox_pred_img = np.array(props_img[0].bbox)
    new_bbox =  (bbox_pred_img[0] -50, bbox_pred_img[1]-50 , bbox_pred_img[2] +50, bbox_pred_img[3]+50 )
    temp_img[new_bbox[0]:new_bbox[2], new_bbox[1]:new_bbox[3]] = 1
    img_final = np.multiply(img, temp_img)
    log_entropy =logsumexp(img_final) / ((props_img[0].bbox_area ))
    return log_entropy


def get_uncertainty_of_all(image_path, all_fnames, pred_path, method ):

    all_uncertainty  =[]
    all_filenames= []
    for i in range(len(all_fnames)):
        uncertainty = get_uncertainty_of_image(image_path +all_fnames[i], pred_path+all_fnames[i])
        all_uncertainty.append(uncertainty)
        all_filenames.append(all_fnames[i][:-4])    #saving without extension png

    all_uncertainty = normalize_list(all_uncertainty)
    df_uncertainty = pd.DataFrame(list(zip(all_filenames, all_uncertainty)),
                   columns =['File_Name', method])

    return df_uncertainty


if __name__=='__main__':

    stage_type = '_ed'          # choices '_ed_' or '_es_'
    method = 'mi'            # method    'atlas' or 'entropy' or 'mi' or 'variance'
    save_csv = 1

    if save_csv:
        pred_path = '/home/lavsen/HDD1/lavsen/results/pred/pred_3_structures_dropout/test_set/'
        fn_path = '/home/lavsen/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set' + stage_type +'.txt'
        base_save_path = '/home/lavsen/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/'
        base_img_path = '/home/lavsen/HDD1/lavsen/results/uncertainty_map/softmax_epochs_100/train_3_structures_dropout/test_set/'

        if method == 'entropy':
            img_path = base_img_path + method + '/'
            save_path = base_save_path + method + '/' + method + stage_type + '.csv'
        elif method == 'variance':
            img_path = base_img_path + method + '/'
            save_path = base_save_path + method + '/' + method + stage_type + '.csv'
        elif method =='mi':
            img_path =base_img_path + method + '/'
            save_path = base_save_path + method + '/' + method + stage_type + '.csv'
        options =[]
        with open(fn_path, 'r') as file:
            for line in file:
                fn = line.strip().split("\n")
                options.append(fn[0] +'.png')
            all_fnames = options
            print (all_fnames)
        df_uncertainty = get_uncertainty_of_all(img_path, all_fnames, pred_path, method )
        df_uncertainty.to_csv(save_path)

    else:
        interval = 10               # interval int - 5, 10 or 20
        graph_type = 'line_plot'     #'box_plot' or 'line_plot'

        save_base_path= '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/'

        if stage_type == '_es' :
            save_path_uncertainty_metric = save_base_path + method + '/' +method  + stage_type +'.csv'
            save_path_gt = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/gt/gt_and_pred_es_ch_2_and_4.csv'
        elif stage_type == '_ed':
            save_path_uncertainty_metric = save_base_path + method + '/' +method  + stage_type +'.csv'
            save_path_gt = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/gt/gt_and_pred_ed_ch_2_and_4.csv'

        df_gt_and_pred = pd.read_csv(save_path_gt)
        df_uncertainty = pd.read_csv(save_path_uncertainty_metric)


        save_path = '/media/HDD1/lavsen/results/graphs/deeplab_dropout_3_structures/'
        save_type = graph_type +'/'+ method + '/' + graph_type + graph_type + '_interval_' + str(interval) +  stage_type + '_' + method +  '.png'
        if graph_type == 'box_plot':
            save_box_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method )
        elif graph_type == 'line_plot':
            save_line_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method)
    

