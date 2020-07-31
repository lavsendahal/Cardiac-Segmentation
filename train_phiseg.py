import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from model_phiseg import PhiSeg
from dataloaders import make_data_loader
#from loaders.loader import TrainDataset, ValDataset
from utils_phiseg import diceOverlap
import argparse

def main():

    parser = argparse.ArgumentParser(description="PyTorch Phiseg Implementation")
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--dataset', type=str, default='camus', choices=['pascal', 'coco', 'cityscapes', 'camus'],
                        help='dataset name (default: camus)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = {
        #'data_path': '../data/',
        'save_path': '../trained_model.pt',
        'log_dir': '../logs/',

        'layer_norm': 'batch_norm',
        'use_logistic_transform': False,
        'latent_levels': 5,
        'resolution_levels': 7,
        'n0': 32,
        'zdim_0': 2,
        'max_channel_power': 4,
        'dimensionality_mode': '2D',
        'image_size': (513, 513),
        'nlabels': 2,
        'num_labels_per_subject': 1,

        'aug_options': {
            'do_flip_lr': True,
            'do_flip_ud': True,
            'do_rotations': True,
            'do_scaleaug': True,
            'nlabels': 2
        },

        'lr_schedule_dict': {0: 1e-3},
        'deep_supervision': True,
        'batch_size': args.batch_size,
        'num_iter': 500000,
        'annotator_range': [0],

        'KL_divergence_loss_weight': 1.0,
        'exponential_weighting': True,
        'residual_multinoulli_loss_weight': 1.0

    }
    print('Loading data')
    #trainset = TrainDataset(root_dir=config['data_path'])
    #trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    #valset = ValDataset(root_dir=config['data_path'])
    #valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)


    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    trainloader, valloader, testloader, nclass, data_size = make_data_loader(args, **kwargs)
            
    data_loaders = {
        'train': trainloader, 'val': valloader
    }

    data_lengths = {
        'train': data_size['train'] / config['batch_size'], 'val': data_size['val']/ config['batch_size']
    }

    print('Constructing Model on Device:', device)
    net = PhiSeg(config, device=device)
    net.to(device)
    net.train(True)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    writer = SummaryWriter(config['log_dir'])

    for epoch in range(100):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
                print('Training')
            else:
                # net.train(False)
                print('Validating')

            running_loss = 0.0
            for i, data in enumerate(data_loaders[phase], 0):
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)

                # zero parameter grads
                optimizer.zero_grad()

                # forward + backward + optimise
                if phase == 'train':
                    loss_total, loss_dict, segmentation = net(inputs, labels, mode=phase)
                    segmentation = F.log_softmax(segmentation, dim=1)
                    segmentation = segmentation.max(dim=1, keepdim=True)[1]
                    loss_total.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        loss_total, loss_dict, segmentation = net(inputs, labels, mode=phase)
                        segmentation = F.log_softmax(segmentation, dim=1)
                        segmentation = segmentation.max(dim=1, keepdim=True)[1]

                running_loss += loss_total.item()

                for key in loss_dict:
                    writer.add_scalar(phase + '/' + key, loss_dict[key], data_lengths[phase] * epoch + i)

                dice = diceOverlap(labels.cpu(), segmentation.cpu())
                writer.add_scalar(phase + '/dice', dice, data_lengths[phase] * epoch + i)
                inp_grid = vutils.make_grid(inputs, normalize=True, scale_each=True)
                lbl_grid = vutils.make_grid(labels)
                pred_grid = vutils.make_grid(segmentation)
                writer.add_image(phase + '/input', inp_grid, data_lengths[phase] * epoch + i)
                writer.add_image(phase + '/label', lbl_grid, data_lengths[phase] * epoch + i)
                writer.add_image(phase + '/pred', pred_grid, data_lengths[phase] * epoch + i)
            writer.add_scalar(phase + '/running_loss', running_loss / data_lengths[phase], epoch)

    print('Finished Training')
    torch.save(net, config['save_path'])
    writer.close()





if __name__ == "__main__":
   main()