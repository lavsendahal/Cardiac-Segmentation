from save_uncertainty_csv import *


if __name__ == '__main__':
    graph_type = 'line_plot'     #'box_plot' or 'line_plot'
    interval =100
    method = 'mi'
    #save_path_uncertainty_metric = '/media/HDD1/lavsen/results/dynamic/csv_files/deeplab_dynamic_dropout/test_set/uncertainty_augment/entropy/entropy_dynamic.csv'
    save_path_uncertainty_metric = '/media/HDD1/lavsen/ouyang/results/csv_files/deeplab_dynamic/uncertainty_dropout/mi/mi_dynamic.csv'
    #save_path_gt = '/media/HDD1/lavsen/results/dynamic/csv_files/deeplab_dynamic_dropout/test_set/gt/gt_and_pred.csv'
    save_path_gt= '/media/HDD1/lavsen/ouyang/results/csv_files/deeplab_dynamic/gt/gt_and_pred.csv' #No dropout
    df_gt_and_pred = pd.read_csv(save_path_gt)
    df_uncertainty = pd.read_csv(save_path_uncertainty_metric)
    save_path = '/media/HDD1/lavsen/ouyang/results/graphs/'
    save_type = 'dropout_mi.png'
    save_line_plot(df_uncertainty, df_gt_and_pred, save_path, save_type, interval, method)
