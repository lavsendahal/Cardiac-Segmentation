import numpy as np
import torch
import os
import argparse 
from torchvision import transforms
from dataloaders import custom_transforms as tr
from scipy.special import softmax
from PIL import Image
from doc.deeplab_resnet_dropout import DeepLabv3_plus
from modeling.sync_batchnorm.replicate import patch_replication_callback

def normalize_img(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img /= 255.0
    img -= mean
    img /= std
    return img


def validation(image_path, f, save_path, model):

    for fn in f:
            #print (fn)
        with torch.no_grad():
            file_path = image_path + fn[:-1] + '.png'
            im = Image.open(file_path).convert('RGB')
            im = np.array(im).astype(np.float32)
            im = normalize_img(im)
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
            im = torch.from_numpy(im).float().to('cuda')
            pred = model(im)
            pred = pred.data.cpu().numpy()
            pred = np.squeeze(pred) 
            pred = softmax(pred, axis=0)
            pred = pred[1,:,:] *255
            pred = Image.fromarray((pred).astype(np.uint8))
            print ('saving at his location', save_path + fn[:-1])
            pred.save(save_path + fn[:-1] +   '.png')
            del pred
            del im
      
            
def create_folders(starting_epoch= None, path = None): 
    for i in range(starting_epoch,starting_epoch+50):
        os.mkdir(path  + 'epoch_' + str (i))
    

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description="Inference for Multiple Epochs")
    parser.add_argument('--is_camus', action='store_true', default=False)
    parser.add_argument('--is_dynamic', action='store_true', default=False)

    
    args = parser.parse_args()

    if args.is_camus:
        fn_path =  '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt'
        save_path = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/HSE/'
        image_path = '/media/HDD1/lavsen/dataset/camus-dataset/test-png/all-images/'
        PATH = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/all_models/camus_dataset/'
        n_classes = 4


    if args.is_dynamic:
        fn_path = '/media/HDD1/lavsen/dataset/dynamic-dataset/ImageSets/test.txt'
        save_path = '/media/HDD1/lavsen/results/dynamic/all_epochs_softmax_pred/deeplab_dynamic_dropout/test_set/'
        image_path = '/media/HDD1/lavsen/dataset/dynamic-dataset/img_dynamic/'
        PATH = '/media/HDD1/lavsen/results/models/deeplab_dynamic_dropout/'
        n_classes=2


    starting_epoch = 50
    #create_folders(starting_epoch =starting_epoch , path = save_path)

    for i in range(0,50):
        f = open(fn_path, "r")
        model = DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=True, freeze_bn=False, _print=True)
        #model = model.to('cuda')
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
        model = model.cuda()
        k = starting_epoch +i
        new_path = PATH + 'epoch_' + str(k) 
        print ('currently evaluating')
        print ('*****************************************************')
        print (new_path)
        print ('*****************************************************')
        assert os.path.isfile(new_path)
        model.load_state_dict(torch.load(new_path))
        model.eval()
        validation(image_path, f, save_path +'epoch_' +str(k) + '/', model)

