import SimpleITK as sitk
import numpy as np
import os
from PIL import Image
import argparse

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


    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--us_view', type=str, default=None,
                        choices=['2_chamber', '4_chamber'],
                        help='Ultrasound view')
    parser.add_argument('--es_or_ed', type=str, default=None,
                        choices=['es', 'ed'],
                        help='End dystolic or End Systolic sequence')
    parser.add_argument('--is_gt', action='store_true', default=False,
                        help='If it is ground truth. Default: False')

    args = parser.parse_args()
	

    if args.us_view == '2_chamber' and args.es_or_ed == 'es' : 
        print ('2 chamber view and es')
        img_type = '_2CH_ES'
        save_type = '_2CH_ES'
    elif args.us_view == '2_chamber' and args.es_or_ed == 'ed' : 
        print ('2 chamber view and ed')

        img_type = '_2CH_ED'
        save_type = '_2CH_ED'

    elif args.us_view == '4_chamber' and args.es_or_ed == 'es' : 
        print ('4 chamber view and es')
        img_type = '_4CH_ES'
        save_type = '_4CH_ES'

    elif args.us_view == '4_chamber' and args.es_or_ed == 'ed' : 
        print ('4 chamber view and ed')
        img_type = '_4CH_ED'
        save_type = '_4CH_ED'
    if args.is_gt:
        print ('it is groundtruth')
        img_type = img_type + '_gt'


    
    file_path = '/home/lavsen/NAAMII/dataset/camus-dataset/testing/'  
    save_path = '/home/lavsen/NAAMII/dataset/camus-dataset/test-png/'

    for folder_names in os.listdir(file_path):
	
        fn = folder_names + img_type + '.mhd'
        img_array, origin, spacing = load_itk(file_path + folder_names + '/' + fn)
        img_array = np.squeeze(img_array)
     


        #print (np.max(img_array))
        #print (np.min(img_array))
        print (img_array.shape)
        im = Image.fromarray(img_array)
        im.save(save_path + img_type +  '/' + folder_names + save_type + '.png')




