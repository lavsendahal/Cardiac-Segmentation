from doc.deeplab_resnet_dropout_ver_1 import DeepLabv3_plus
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr
from scipy.special import softmax


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
                #pred = np.argmax(pred, axis=1)
            pred = np.squeeze(pred) 
            pred = softmax(pred, axis=0)
            pred = pred[1,:,:] *255
       	  		                
		#pred[pred ==1] = 255
                #pred[pred ==2] = 170
                #pred[pred ==3] = 255
            pred = Image.fromarray((pred).astype(np.uint8))
          
            print ('saving at his location', save_path + fn[:-1])
            pred.save(save_path + fn[:-1] +   '.png')
            del pred
            del im
      
            


if __name__ == '__main__':
 
    fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt'
    save_path_all = '/media/HDD1/lavsen/results/all_epochs_softmax_pred/deeplab_lv_both_all_3_struct/test_set/'
    image_path = '/media/HDD1/lavsen/dataset/camus-dataset/test-png/all-images/'
   
    PATH = '/media/HDD1/lavsen/results/models/deeplab_lv_both_3_structures_dropout_ver1/'
   
    starting_epoch = 430
    for i in range(0,200):
        f = open(fn_path, "r")
        model = DeepLabv3_plus(nInputChannels=3, n_classes=4, os=16, pretrained=True, freeze_bn=False, _print=True)
        model = torch.nn.DataParallel(model).cuda()
        k = starting_epoch +i
        new_path = PATH + 'epoch_' + str(k) 
        print ('currently evaluating')
        print ('*****************************************************')
        print (new_path)
        print ('*****************************************************')
        model.load_state_dict(torch.load(new_path))
        model.eval()
        new_save_path = save_path_all + 'epoch_' + str(k) + '/'
        print ('save path is',new_save_path)
        validation(image_path, f, new_save_path, model)

    
  
    
    #for key in new_checkpoint:
        #print(key)

    #print (model)
    #model.load_state_dict(torch.load(PATH))
    #model.load_state_dict(net)ls
    
    
    #print ('********************')
    #print (model)
    #checkpoint = torch.load(PATH)
    #print (checkpoint.state_dict())
    #new_checkpoint = OrderedDict()
    #model.load_state_dict(checkpoint['state_dict'])
    #for key in checkpoint:
     #   print(key)
    
    
    #model = model.to('cuda')
  
