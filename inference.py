from doc.deeplab_resnet import DeepLabv3_plus
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr



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
             #   print ('pred shape', pred.shape)
                pred = np.argmax(pred, axis=1)
                pred = np.squeeze(pred)
                pred[pred ==1] = 255
                #pred[pred ==2] = 170
                #pred[pred ==3] = 255
                pred = Image.fromarray((pred).astype(np.uint8))
                print ('saving at his location', save_path + fn[:-1])
                pred.save(save_path + fn[:-1] +   '.png')
                del pred
                del im
      
             


if __name__ == '__main__':
 
    fn_path = '/home/lavsen/NAAMII/dataset/camus-dataset/ImageSets/Segmentation_camus/val.txt'
    save_path_all = '/home/lavsen/NAAMII/results/uncertainty/'
    image_path = '/home/lavsen/NAAMII/dataset/camus-dataset/all_images/'
   
    PATH = '/home/lavsen/NAAMII/results/camus_resnet_lv_allmodels/'
   
    starting_epoch = 130
    for i in range(0,50):
        f = open(fn_path, "r")
        model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True, freeze_bn=False, _print=True)
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
    #model.load_state_dict(net)
    
    #print ('********************')
    #print (model)
    #checkpoint = torch.load(PATH)
    #print (checkpoint.state_dict())
    #new_checkpoint = OrderedDict()
    #model.load_state_dict(checkpoint['state_dict'])
    #for key in checkpoint:
     #   print(key)
    
    
    #model = model.to('cuda')
  