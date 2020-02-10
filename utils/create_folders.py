import os


if __name__ =='__main__':
    
    path = '/home/lavsen/NAAMII/results/uncertainty/'
    starting_epoch = 130 
    for i in range(0,50):
        os.mkdir(path + 'epoch_' + str (starting_epoch +(i)))
    