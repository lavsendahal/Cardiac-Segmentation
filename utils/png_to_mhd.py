import argparse
import csv
import pandas as pd
import os
from PIL import Image
import numpy as np
import SimpleITK as sitk

def load_itk(mhd_fn):
    '''
    This function reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''

    img = sitk.ReadImage(mhd_fn)
    img_array = sitk.GetArrayFromImage(img)
    direction = np.array(list(reversed( img.GetDirection())))
    #origin = np.array(list(reversed(img.GetOrigin())))
    #spacing = np.array(list(reversed(img.GetSpacing())))
    origin = img.GetOrigin()
    spacing = img.GetSpacing()


    return img_array, origin, spacing, direction


def text_to_csv(PATH):
    temp_list = []
    f = open(PATH+ 'val_all_images.txt', "r")
    for line in f:
        temp_list.append(line[:-1])

    df = pd.DataFrame(temp_list)
    print (df)
    df.to_csv(PATH + 'val_all_images.csv', sep='\t')


def png_to_mhd(PATH, mhd_path, png_path, save_path):
    f = open(PATH, "r")
    all_files = os.listdir(mhd_path)
    for line in f:
        mhd_fn = mhd_path + line[0:11] + '/' + line[:-1] + '.mhd'
        print (mhd_fn)
        img_array, origin, spacing, direction = load_itk(mhd_fn)
        print ('direction is', direction)
        print ('origin is', origin)
        print ('spacing is' , spacing)
        png_fn = png_path + '/' + line[:-1] + '.png'
        im = Image.open(png_fn)
        im_array = np.array(im)
        img_sitk = sitk.GetImageFromArray(im_array)
        img_sitk.SetOrigin(origin)
        img_sitk.SetSpacing(spacing)
        #img_sitk.SetDirection(direction)

        save_name = save_path + line[:-1]
        print (save_name)
        sitk.WriteImage(img_sitk, save_name + '.mhd')
        #print (im_array.shape)


if __name__ == '__main__':
    fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/Segmentation_camus/val.txt'
    mhd_path = '/media/HDD1/lavsen/dataset/camus-dataset/training/'
    png_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/pred/val_set/'
    save_path = '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/pred/val_set_mhd/'
    #text_to_csv(PATH)
    png_to_mhd(fn_path, mhd_path, png_path, save_path)
