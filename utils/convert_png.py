import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

def load_itk(filename):
    '''
    This function reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''

    img = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(img)
    origin = np.array(list(reversed(img.GetOrigin())))
    spacing = np.array(list(reversed(img.GetSpacing())))

    return img_array, origin, spacing

if __name__ == "__main__":
    file_path = '/home/lavsen/NAAMII/dataset/camus-dataset/training/'
    save_path = '/home/lavsen/NAAMII/dataset/camus-dataset/'
    img_type = '_2CH_ED_gt'
    #img_type = '_2CH_ED'
    save_type = '_2CH_ED'

    for folder_names in os.listdir(file_path):

        fn = folder_names + img_type + '.mhd'
        img_array, origin, spacing = load_itk(file_path + folder_names + '/' + fn)
        img_array = np.squeeze(img_array)
        #img_array[img_array == 1] = 85
        #img_array[img_array == 2] = 170
        #img_array[img_array == 3] = 255


        print (np.max(img_array))
        print (np.min(img_array))
        print (img_array.shape)
        im = Image.fromarray(img_array)
        im.save(save_path + img_type +  '/' + folder_names + save_type + '.png')




