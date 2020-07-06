import argparse 
import torch
from doc.deeplab_resnet_dropout import DeepLabv3_plus
from dataloaders import make_data_loader
from tqdm import tqdm    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference For Best Model")
    parser.add_argument('--dataset', type=str, default='camus', choices=['pascal', 'camus'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=513, help='crop image size')
    args = parser.parse_args()

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    num_img_tr = len(test_loader)
    print ('total no for dataloader in loop ', num_img_tr)
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4, os=16, pretrained=False, freeze_bn=False, _print=True)   
    PATH = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar' 
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to('cuda')
    model.eval()

    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
            pred = output.data.cpu().numpy()    
            print ('the size of pred array is ', pred.shape)