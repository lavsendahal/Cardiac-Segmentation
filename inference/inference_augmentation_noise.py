from doc.deeplab_resnet_dropout_0 import DeepLabv3_plus
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
from torchvision import transforms
from scipy.special import softmax
import random
import os
import argparse 

def normalize_img(img):
   
      mean = (0.485, 0.456, 0.406)
      std = (0.229, 0.224, 0.225)
      img /= 255.0
      img -= mean
      img /= std
      return img

def validation(image_path, f, save_path, model):

        for fn in f:
            print (fn)
            with torch.no_grad():
                file_path = image_path + fn[:-1] + '.png'
                im = Image.open(file_path).convert('RGB')
                rotate_degree = random.uniform(-1*20, 20)
                if random.random() < 0.5:
                        im = im.filter(ImageFilter.GaussianBlur(radius=random.random()))
                flip_prob = random.random()
                if flip_prob < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                
                im = im.rotate(rotate_degree, Image.BILINEAR)
                im = np.array(im).astype(np.float32)
                im = normalize_img(im)
                im = im.transpose((2, 0, 1))
                im = np.expand_dims(im, axis = 0)
                im = torch.from_numpy(im).float().to('cuda')
                pred = model(im)
                pred = pred.data.cpu().numpy()
                print ('pred shape', pred.shape)
                pred = np.squeeze(pred)
                pred = softmax(pred, axis=0)
                pred = pred[1,:,:] *255
                pred_img =  Image.fromarray((pred).astype(np.uint8))
                pred_img = pred_img.rotate(rotate_degree * -1, Image.BILINEAR)
                if flip_prob < 0.5:
                    pred_img = pred_img.transpose(Image.FLIP_LEFT_RIGHT)
                #pred = np.argmax(pred, axis=1)
                #pred[pred ==1] = 255
                #pred[pred ==2] = 170
                #pred[pred ==3] = 255
                #pred = Image.fromarray((pred).astype(np.uint8))
                pred_img.save(save_path + fn[:-1] +   '.png')
                del pred
                del im
  
def create_folders(starting_index= None, path = None): 
    for i in range(0,100):
        os.mkdir(path + 'augment_' + str (starting_index +(i)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference for Augmentation")
    parser.add_argument('--is_camus', action='store_true', default=False)
    parser.add_argument('--is_dynamic', action='store_true', default=False)
    args = parser.parse_args()

    if args.is_camus:
        fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/Segmentation_camus/val_2ch.txt'
        image_path = '/media/HDD1/lavsen/dataset/camus-dataset/all_images/'  
        save_path =  '/media/HDD1/lavsen/results/camus/deeplab_camus_no_dropout/all_augment_softmax_pred/val_set/'
        PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3_no_dropout_size_513/model_best.pth.tar'
        n_classes = 4


    if args.is_dynamic:
        fn_path = '/media/HDD1/lavsen/dataset/dynamic-dataset/ImageSets/test.txt'
        save_path = '/media/HDD1/lavsen/results/dynamic/all_augment_softmax_pred/deeplab_dynamic/test_set/'
        image_path = '/media/HDD1/lavsen/dataset/dynamic-dataset/img_dynamic/'
        PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/dynamic/deeplabv3plus-dynamic/model_best.pth.tar'
        n_classes = 2
        
    starting_index = 0
    #create_folders(starting_index, save_path)
    
    for augment_count in range(100):
        f = open(fn_path, "r")
        model = DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=False, freeze_bn=False, _print=True)
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        model.eval()
        validation(image_path, f, save_path +'augment_' +str(augment_count) + '/', model)
        f.close()