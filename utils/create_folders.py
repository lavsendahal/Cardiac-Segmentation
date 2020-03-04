import os


if __name__ =='__main__':
    
    path = '/media/HDD1/lavsen/results/all_epochs_softmax_pred/deeplab_lv_both_all_3_struct/test_set/'
    starting_epoch = 300
    for i in range(0,200):
        os.mkdir(path + 'epoch_' + str (starting_epoch +(i)))
        #os.mkdir(path + 'results_' + str ((i)))
    
