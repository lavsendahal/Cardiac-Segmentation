from doc.deeplab_resnet_dropout import DeepLabv3_plus
from PIL import Image, ImageFilter
import numpy as np
import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr
import random

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
                im = im.filter(ImageFilter.GaussianBlur(radius=random.random()))
                im = np.array(im).astype(np.float32)
                im = normalize_img(im)
                im = im.transpose((2, 0, 1))
                im = np.expand_dims(im, axis = 0)
                im = torch.from_numpy(im).float().to('cuda')
                pred = model(im)
                pred = pred.data.cpu().numpy()
                print ('pred shape', pred.shape)
                pred = np.argmax(pred, axis=1)
                pred = np.squeeze(pred)
                pred[pred ==1] = 255
                #pred[pred ==2] = 170
                #pred[pred ==3] = 255
                pred = Image.fromarray((pred).astype(np.uint8))
                pred.save(save_path + fn[:-1] +   '.png')
                del pred
                del im




if __name__ == '__main__':
 
    fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/Segmentation_camus/val.txt'
    save_path = '/media/HDD1/lavsen/results/pred/pred_lv_both_stage_augmentation/val_set/'
    image_path = '/media/HDD1/lavsen/dataset/camus-dataset/all_images/'
    PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus_lv/deeplabv3plus-lv-resnet_dropout/model_best.pth.tar'
    
    for dropout_count in range(100):
        f = open(fn_path, "r")
        model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=False, freeze_bn=False, _print=True)
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        model.eval()
        validation(image_path, f, save_path +'results_' +str(dropout_count) + '/', model)
        f.close()
        