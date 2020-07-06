import argparse 
import numpy as np
import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr
from doc.deeplab_resnet_dropout import DeepLabv3_plus
from PIL import Image


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
                # print (file_path)
                im = Image.open(file_path).convert('RGB')
                im = np.array(im).astype(np.float32)
                # print ('largest before normalize', np.max(im))
                im = normalize_img(im)
                im = im.transpose((2, 0, 1))
                im = np.expand_dims(im, axis = 0)
                # print ('largest value', np.max(im))
                # print ('min value', np.min(im))
                im = torch.from_numpy(im).float().to('cuda')
                pred = model(im)
                pred = pred.data.cpu().numpy()
                print ('pred shape', pred.shape)
                
                pred = np.argmax(pred, axis=1)
                pred = np.squeeze(pred)
                print('largest value in prediction', np.max(pred))
                #pred= pred*255
                #pred[pred ==1] = 85
                #pred[pred ==2] = 170
                #pred[pred ==3] = 255
                pred = Image.fromarray((pred).astype(np.uint8))
                pred.save(save_path + fn[:-1] +   '.png')
                del pred
                del im
            
             

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference For Best Model")
    parser.add_argument('--is_camus', action='store_true', default=False)
    parser.add_argument('--is_dynamic', action='store_true', default=False)
    args = parser.parse_args()

    if args.is_camus:
        fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt'
        image_path = '/media/HDD1/lavsen/dataset/camus-dataset/test-png/all-images/'
        save_path = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/pred_best_model/test_set/'
        PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar'
   
    if args.is_dynamic:
        fn_path = '/media/HDD1/lavsen/ouyang/dataset/ImageSets/test.txt'
        save_path = '/media/HDD1/lavsen/ouyang/results/pred/model_trained_on_camus_data/test_set/'
        image_path = '/media/HDD1/lavsen/ouyang/dataset/test_set/images/'
        PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus_lv_ed/deeplabv3plus-lv-ed/model_best.pth.tar'
        #PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus_lv/deeplabv3plus-lv-resnet_dropout_ver1/model_best.pth.tar'
    
    f = open(fn_path, "r")
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4, os=16, pretrained=False, freeze_bn=False, _print=True)
    #model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint)   #USE THE ABOVE LINE TO PREDICT FOR BEST MODEL
    model = model.to('cuda')
    
    model.eval()
    validation(image_path, f, save_path, model)