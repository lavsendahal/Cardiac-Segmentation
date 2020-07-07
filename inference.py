import argparse 
import torch
import numpy as np
import torchvision
from doc.deeplab_resnet_dropout import DeepLabv3_plus
from dataloaders import make_data_loader
from tqdm import tqdm    
from PIL import Image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference For Best Model")
    parser.add_argument('--dataset', type=str, default='camus', choices=['pascal', 'camus'])
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    
    BEST_MODEL_PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar' 
    SAVE_PATH = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/pred_best_model/test_set/'
    test_fn_path = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt' 
    test_fn = open(test_fn_path, 'r')
    test_fn_list = [line.split() for line in test_fn.readlines()]

    # Inference for best Model
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4, os=16, pretrained=False, freeze_bn=False, _print=True)   
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to('cuda')
    model.eval()
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']            
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
                output = output.data.cpu().numpy()  
            pred = output[0,:,:,:]
            pred = np.argmax(pred, axis=0)

            # pred[pred ==1] = 85 ; pred[pred ==2] = 170 ; pred[pred ==3] = 255 #Only for Visualization
            pred = Image.fromarray((pred).astype(np.uint8))
            pred.save(SAVE_PATH + test_fn_list[i][0] +   '.png')

