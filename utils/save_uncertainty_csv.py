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
from sklearn.metrics import jaccard_score
import argparse 




def combine_both_stages_dice(df):
    name_col = df.loc[ : , 'File_Name' ]
    dice_col= df.loc[ : , method ]
    dice_patient = []
    for i in range(50):
        dice_patient.append((dice_col[i] + dice_col[i+50])/2 )
    all_names = []
    name_type = 'patient'
    for i in range(50):
        new_str = name_type + str('000') + str(i)
        all_names.append(name_col[i][:-7])

    df_dice_patients = pd.DataFrame(list(zip(all_names, dice_patient)),
                    columns =['File_Name',method])
    return df_dice_patients


def combine_both_stages_dice_log_scale(df):
    name_col = df.loc[ : , 'File_Name' ]
    dice_col= df.loc[ : , method ]
    dice_patient = []
    for i in range(50):
        dice_patient.append(logsumexp([dice_col[i] , dice_col[i+50]]))
    all_names = []
    name_type = 'patient'
    for i in range(50):
        new_str = name_type + str('000') + str(i)
        all_names.append(name_col[i][:-7])

    df_dice_patients = pd.DataFrame(list(zip(all_names, dice_patient)),
                    columns =['File_Name',method])
    return df_dice_patients


def normalize_list(given_list):
    '''
    Given a list, returns the max-min normalized list.
    '''
    amin, amax = min(given_list), max(given_list)
    for i, val in enumerate(given_list):
        given_list[i] = (val-amin) / (amax-amin)
    return given_list

def calculate_dice_all(atlas_path, all_fnames, pred_path, method):
    'Returns the dataframe with filenames and the corresponding dice score and average dice score'

    all_uncertainty  =[]
    all_filenames= []
    for i in range(len(all_fnames)):
        uncertainty = calculate_dsc(atlas_path +all_fnames[i], pred_path+all_fnames[i])
        all_uncertainty.append(uncertainty)
        all_filenames.append(all_fnames[i][:-4])    #saving without extension png

    df_uncertainty = pd.DataFrame(list(zip(all_filenames, all_uncertainty)),
                   columns =['File_Name', method])

    return df_uncertainty

def calculate_dsc(gt_path, pred_path ):
    '''
    gt_path - Ground Truth Absolute path of Image
    pred_path - Prediction Absolute path of Image
    Loads the file from the two specified paths and computes dice coefficient

    '''
    im_pred = Image.open(pred_path)
    im_gt = Image.open(gt_path)
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


def get_data_for_box_plot(df_uncertainty, df_gt_and_pred, save_path, interval, uncertainty_type):
    '''
    atlas_and_pred_dice - A pandas dataframe for atlas and pred -  dice
    gt_and_pred_dice - A pandas dataframe for gt and pred - dice
    save_path - Path where the box plots gets saved
    interval - integer (int) for different intervals of samples to plot box plot.
    Saves the box plot for different no of samples intervals.
    Returns the pandas dataframe containing dice of different intervals given by list x and corresponding dice coefficient.
    '''
    #uncertainty_type='Dice_Coefficient'

    sorted_df_uncertainty = df_uncertainty.sort_values(by=uncertainty_type, ascending=True)
   
    if uncertainty_type =='atlas':
        sorted_df_uncertainty = df_uncertainty.sort_values(by=uncertainty_type, ascending=False)
    sorted_df_uncertainty_file_list = sorted_df_uncertainty['File_Name'].tolist()
    all_dice = []
    y =[]
    x=[]

    for i in range(interval,len(sorted_df_uncertainty_file_list) + 1,interval):
        current_list = sorted_df_uncertainty_file_list[0:i]
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
    print (image_path)
    print (pred_path)
    img = np.array(Image.open(image_path)).astype(np.float32)
    temp_img = np.zeros((img.shape[0], img.shape[1]))

    img_pred = np.array(Image.open(pred_path)).astype(int)
    props_img = regionprops(img_pred)
    #print (props_img)
    if len(props_img) >0 :
        bbox_pred_img = np.array(props_img[0].bbox)
        new_bbox = (bbox_pred_img[0], bbox_pred_img[1] , bbox_pred_img[2], bbox_pred_img[3] )
    #new_bbox =  (bbox_pred_img[0] -50, bbox_pred_img[1]-50 , bbox_pred_img[2] +50, bbox_pred_img[3]+50 )
        temp_img[new_bbox[0]:new_bbox[2], new_bbox[1]:new_bbox[3]] = 1
        img_final = np.multiply(img, temp_img)
        log_entropy =logsumexp(img_final) / ((props_img[0].bbox_area ))
    else:
        img_final = np.multiply(img, img_pred)
        log_entropy =logsumexp(img_final) / (img.shape[0]* img.shape[1])
    # img_final = np.multiply(img,img_pred)
    # log_entropy =logsumexp(img_final) / ((img.shape[0]* img.shape[1] ))

    return log_entropy


def get_uncertainty_of_all(image_path, all_fnames, pred_path, method ):

    all_uncertainty  =[]
    all_filenames= []
    for i in range(len(all_fnames)):
        uncertainty = get_uncertainty_of_image(image_path +all_fnames[i], pred_path+all_fnames[i])
        all_uncertainty.append(uncertainty)
        print (all_fnames[i][:-4])
        print ('********************we are here')
        all_filenames.append(all_fnames[i][:-4])    #saving without extension png

    all_uncertainty = normalize_list(all_uncertainty)
    df_uncertainty = pd.DataFrame(list(zip(all_filenames, all_uncertainty)),
                   columns =['File_Name', method])

    return df_uncertainty


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Saving Uncertainty Scores csv files and graphs")
    parser.add_argument('--is_camus', action='store_true', default=False)
    parser.add_argument('--is_dynamic', action='store_true', default=False)

    parser.add_argument('--is_epoch', action='store_true', default=False)
    parser.add_argument('--is_dropout', action='store_true', default=False)
    parser.add_argument('--is_augment', action='store_true', default=False)

    parser.add_argument('--is_variance', action='store_true', default=False)
    parser.add_argument('--is_entropy', action='store_true', default=False)
    parser.add_argument('--is_mi', action='store_true', default=False)
    parser.add_argument('--is_atlas', action='store_true', default=False)

    parser.add_argument('--is_ed', action='store_true', default=False)
    parser.add_argument('--is_es', action='store_true', default=False)

    parser.add_argument('--threshold_atlas', type=float, default=None)

    args = parser.parse_args()


    if args.is_camus:
        if args.is_ed:
            stage_type = '_ed'
        if args.is_es:
            stage_type = '_es'
        if args.is_epoch:
            base_save_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/csv_files/test_set/uncertainty_epochs/'
            base_img_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/test_set/softmax_epochs/'
        
        if args.is_dropout:
            base_save_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/csv_files/test_set/uncertainty_dropout/'
            base_img_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/test_set/softmax_dropout/'

        if args.is_augment:
            base_save_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/csv_files/test_set/uncertainty_augment/'
            base_img_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/uncertainty_images/test_set/softmax_augment/'
        
        #fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/val_set' + stage_type +'.txt'
        fn_path = '/home/lavsen/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set' + stage_type +'.txt'
        #pred_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/pred/val_set/'
        pred_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/pred/test_set/'
        dataset = '_camus'


    if args.is_dynamic:
        if args.is_epoch:
            base_save_path = '/media/HDD1/lavsen/ouyang/results/csv_files/deeplab_dynamic/uncertainty_epoch/'
            base_img_path = '/media/HDD1/lavsen/ouyang/results/uncertainty_images/softmax_epochs/test_set/'
            
        if args.is_dropout:
            base_save_path = '/media/HDD1/lavsen/ouyang/results/csv_files/deeplab_dynamic/uncertainty_dropout/'
            base_img_path = '/media/HDD1/lavsen/ouyang/results/uncertainty_images/softmax_dropout/test_set/'
            
        if args.is_augment:
            base_save_path = '/media/HDD1/lavsen/ouyang/results/csv_files/deeplab_dynamic/uncertainty_augment/'
            base_img_path = '/media/HDD1/lavsen/ouyang/results/uncertainty_images/softmax_augment/test_set/'
            
        fn_path = '/media/HDD1/lavsen/ouyang/dataset/ImageSets/test.txt'
        pred_path = '/media/HDD1/lavsen/ouyang/results/pred/test_set/'
        dataset = '_dynamic_'
        stage_type = '_both'


    if args.is_entropy:
        method = 'entropy'
        img_path = os.path.join(base_img_path, method + '/')
        save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type +'.csv')
        
    elif args.is_variance:
        method = 'variance'
        img_path = os.path.join(base_img_path, method + '/')
        save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type +'.csv')
       
    elif args.is_mi:
        method= 'mi'
        img_path = os.path.join(base_img_path, method + '/')
        save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + '.csv')
        
    elif args.is_atlas:
        method = 'atlas'
        if args.threshold_atlas ==0.1:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + 'threshold_0_1.csv')
            atlas_path = base_img_path + 'atlas_thresholded_0_1/'
        if args.threshold_atlas ==0.2:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + 'threshold_0_2.csv') 
            atlas_path = base_img_path + 'atlas_thresholded_0_2/'
        if args.threshold_atlas ==0.3:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset +stage_type + 'threshold_0_3.csv')  
            atlas_path = base_img_path + 'atlas_thresholded_0_3/'

        if args.threshold_atlas ==0.5:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + 'threshold_0_5.csv')
            atlas_path = base_img_path + 'atlas_thresholded_0_5/'

        if args.threshold_atlas ==0.7:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + 'threshold_0_7.csv')
            atlas_path = base_img_path + 'atlas_thresholded_0_7/'
        if args.threshold_atlas ==0.8:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset + stage_type + 'threshold_0_8.csv') 
            atlas_path = base_img_path + 'atlas_thresholded_0_8/'
        if args.threshold_atlas ==0.9:
            img_path = os.path.join(base_img_path, method + '/')
            save_path = os.path.join(base_save_path , method + '/' , method + dataset +stage_type + 'threshold_0_9.csv')  
            atlas_path = base_img_path + 'atlas_thresholded_0_9/'

    options =[]
    with open(fn_path, 'r') as file:
        for line in file:
            fn = line.strip().split("\n")
            options.append(fn[0] +'.png')
        all_fnames = options
        print (all_fnames)
    if args.is_atlas:
        df_uncertainty = calculate_dice_all(atlas_path, all_fnames, pred_path, method)
        print (df_uncertainty.head())
        # df_combined = combine_both_stages_dice_log_scale(df_uncertainty)
        # df_combined.to_csv(save_path[:-4] + '_combined_log' + '.csv')
        if args.is_camus:
            df_combined = combine_both_stages_dice(df_uncertainty)
            df_combined.to_csv(save_path[:-4] + '_combined' + '.csv')
        df_uncertainty.to_csv(save_path)
        
    else:
        df_uncertainty = get_uncertainty_of_all(img_path, all_fnames, pred_path, method )
        print (df_uncertainty.head())
        # df_combined = combine_both_stages_dice_log_scale(df_uncertainty)
        # df_combined.to_csv(save_path[:-4] + '_combined_log' + '.csv')
        if args.is_camus:
            df_combined = combine_both_stages_dice(df_uncertainty)
            df_combined.to_csv(save_path[:-4] + '_combined' + '.csv')
        df_uncertainty.to_csv(save_path)


    # else:
    #     interval = 10               # interval int - 5, 10 or 20
    #     graph_type = 'box_plot'     #'box_plot' or 'line_plot'

    #     if stage_type == '_es' :
    #         save_path_uncertainty_metric = base_save_path + method + '/' +method  + stage_type +'_both.csv'
    #         save_path_gt = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/gt/test_set/gt_and_pred_es_ch_2_and_4.csv'
    #     elif stage_type == '_ed':
    #         save_path_uncertainty_metric = base_save_path + method + '/' +method  + stage_type +'_both.csv'
    #         save_path_gt = '/media/HDD1/lavsen/results/csv_files/deeplab_lv_train_both_3_structures/gt/test_set/gt_and_pred_ed_ch_2_and_4.csv'

    #     df_gt_and_pred = pd.read_csv(save_path_gt)
    #     df_uncertainty = pd.read_csv(save_path_uncertainty_metric)


    #     save_path = '/media/HDD1/lavsen/results/graphs/deeplab_dropout_3_structures/'
    #     save_type = graph_type +'/'+ uncertainty_type +  method + '/' + graph_type + graph_type + '_interval_' + str(interval) +  stage_type + '_' + method +  '.png'
    #     if graph_type == 'box_plot':
    #         save_box_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method )
    #     elif graph_type == 'line_plot':
    #         save_line_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method)
    

