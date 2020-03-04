import numpy as np
import pandas as pd
import os
from sklearn.metrics import jaccard_score
from PIL import Image
import matplotlib.pyplot as plt



def get_data_for_box_plot(atlas_and_pred_dice, df_gt_and_pred, save_path, interval):
    '''
    atlas_and_pred_dice - A pandas dataframe for atlas and pred -  dice
    gt_and_pred_dice - A pandas dataframe for gt and pred - dice
    save_path - Path where the box plots gets saved
    interval - integer (int) for different intervals of samples to plot box plot.
    Saves the box plot for different no of samples intervals.
    Returns the pandas dataframe containing dice of different intervals given by list x.
    '''
    sorted_df_atlas_and_pred = atlas_and_pred_dice.sort_values(by='Dice_Coefficient', ascending=False)
    sorted_df_atlas_and_pred.head()
    sorted_df_atlas_and_pred_file_list = sorted_df_atlas_and_pred['File_Name'].tolist()
    all_dice = []
    y =[]
    x=[]

    for i in range(interval,105,interval):
        current_list = sorted_df_atlas_and_pred_file_list[0:i]
        dsc=[]
        for filenames in current_list:
            array = [filenames]
            dice_coef =  (df_gt_and_pred.loc[df_gt_and_pred['File_Name'].isin(array)])['Dice_Coefficient'].values
            dsc.append(dice_coef[0])
        all_dice.append(dsc)
        print ('Average Dice', sum(dsc)/len(dsc))
        y.append(sum(dsc)/len(dsc))
        x.append(str(i))

    return pd.DataFrame(all_dice), x, y

def save_box_plot(atlas_and_pred_dice, gt_and_pred_dice, save_path, save_type, interval):

    df, x, y = get_data_for_box_plot(atlas_and_pred_dice, gt_and_pred_dice, save_path, interval)
    df1_transposed = df.transpose()
    df1_transposed.columns =x
    _=df1_transposed.boxplot()
    plt.xlabel('No of Samples')
    plt.ylabel('DSC')
    plt.show()
    plt.savefig(save_path + save_type)


def save_line_plot(atlas_and_pred_dice, gt_and_pred_dice, save_path, save_type, interval):
    _, x, y = get_data_for_box_plot(atlas_and_pred_dice, gt_and_pred_dice, save_path, interval)
    x = [int(i) for i in x]
    plt.xlabel('No of Samples')
    plt.ylabel('DSC')
    plt.plot(x, y)
    plt.savefig(save_path + save_type)


def calculate_dice_all(fn_path, gt_path, pred_path):
    'Returns the dataframe with filenames and the corresponding dice score and average dice score'
    f = open(fn_path, "r")
    #pred_path = pred_path + current_pred
    js_sum =0
    count=1
    dice_coeff =[]
    file_names = []
    i =0
    for fn in f:
        dice =  calculate_dsc(fn[:-1], gt_path, pred_path )
        dice_coeff.append(dice)
        file_names.append(fn[:-1])
        count = count+1
        i = i+1
    print ('total files are', count-1)
    print ('average dice coefficient', sum(dice_coeff)/len(dice_coeff) )
    df_dice = pd.DataFrame(list(zip(file_names, dice_coeff)),
                   columns =['File_Name', 'Dice_Coefficient'])
    return df_dice

def calculate_dsc(file_name, gt_path, pred_path ):
    '''
    file_name - Name of file
    gt_path - Ground Truth Absolute path
    pred_path - Prediction Absolute path
    Loads the file from the two specified paths and computes dice coefficient

    '''
    gt_file = gt_path + file_name + '.png'
    pred_file = pred_path + file_name + '.png'
    im_pred = Image.open(pred_file)
    im_gt = Image.open(gt_file)
    im_pred = np.array(im_pred).astype(np.float32)
    im_pred[im_pred==2] = 0
    im_pred[im_pred==3]= 0 
    im_gt =np.array(im_gt).astype(np.float32)
    if np.max(im_pred) >1:
        im_pred = im_pred / 255.0
        
        
        
    if np.max(im_gt) >1:
        im_gt = im_gt / 255.0
    js = jaccard_score(im_gt.flatten(), im_pred.flatten())
    dice_coeff = js*2/(1+js)
    return dice_coeff




if __name__=='__main__':

    pred_path = '/media/HDD1/lavsen/results/pred/pred_3_structures_dropout/test_set/'
    atlas_path = '/media/HDD1/lavsen/results/uncertainty_map/softmax_epochs_100/train_3_structures_dropout/test_set/atlas_thresholded_0_1/'    #Thresholded at 0.3 to create mask
    #gt_path = '/media/HDD1/lavsen/dataset/camus-dataset/all_images_gt_lv/'
    fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set_es.txt'
    save_path = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/atlas_and_pred_0_1_es.csv'

    df_dice = calculate_dice_all(fn_path, atlas_path, pred_path)
    df_dice.to_csv(save_path)

    #
    # stage_type = '_es_'          # choices '_ed_' or '_es_'
    # method = 'atlas'            # method    'atlas' or 'entropy' or 'mutual_information' or 'variance'
    # interval = 20               # interval int - 5, 10 or 20
    # graph_type = 'box_plot'     #'box_plot' or 'line_plot'
    #
    #
    #
    #
    # if stage_type == '_es_' :
    #     base_path = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_es/'
    #     dice_save_path_atlas = base_path + 'atlas_and_pred_lv_train_es_val.csv'
    #     dice_save_path_gt =  base_path + 'gt_and_pred_lv_train_es_val.csv'
    #     train_type = 'deeplab_lv_train_es/'
    # elif stage_type == '_ed_':
    #     base_path = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_ed/'
    #     dice_save_path_atlas = base_path + 'atlas_and_pred_lv_train_ed_val.csv'
    #     dice_save_path_gt = base_path + 'gt_and_pred_lv_train_ed_val.csv'
    #     train_type = 'deeplab_lv_train_ed/'
    #
    # df_gt_and_pred = pd.read_csv(dice_save_path_gt)
    # df_atlas_and_pred = pd.read_csv(dice_save_path_atlas)
    #
    #
    # save_path = '/media/HDD1/lavsen/results/graphs/'
    # save_type = graph_type +'/'+ train_type + method + '/' + graph_type + '_interval_' + str(interval) +  stage_type + method +  '.png'
    # if graph_type == 'box_plot':
    #     save_box_plot(df_atlas_and_pred, df_gt_and_pred, save_path, save_type, interval)
    # elif graph_type == 'line_plot':
    #     save_line_plot(df_atlas_and_pred, df_gt_and_pred, save_path, save_type, interval)
    #
    #
