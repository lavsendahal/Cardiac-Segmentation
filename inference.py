import argparse 
import torch
import numpy as np
import os
import torchvision
from doc.deeplab_resnet_dropout import DeepLabv3_plus
from dataloaders import make_data_loader
from tqdm import tqdm    
from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback

def enable_dropout(m):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

def save_output_images(LOAD_PATH_MODEL, model, test_loader, SAVE_PATH_MODEL_OUTPUT, test_fn_list, sampling_strategy= None):
    '''Saves the prediction image from segmentation model in local disk. Supports only batch size 1 now.
    Args:
        LOAD_PATH_MODEL: Path where model weights are saved
        test_loader: dataloader
        SAVE_PATH_MODEL_OUTPUT: save path for segmentation outputs
        test_fn_list : List of filenames in test set
        sampling_strategy : Choice ['hse', 'dropout', 'augment', 'None']
    '''
    # To Do Pytorch load_state_dict error in multi-gpu use due to different style of saving for best model and all models
    if args.sampling_strategy == 'hse':      
        model.load_state_dict(torch.load(LOAD_PATH_MODEL))   
    else:
        checkpoint = torch.load(LOAD_PATH_MODEL)
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
            pred.save(SAVE_PATH_MODEL_OUTPUT + test_fn_list[i][0] +   '.png')
    return None

def create_folders(starting_sample= None, path = None, no_samples=50): 
    for i in range(starting_sample,starting_sample+ no_samples):
        os.mkdir(path + 'sample_' + str (i))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference For Best Model")
    parser.add_argument('--dataset', type=str, default='camus', choices=['pascal', 'camus'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sampling_strategy', type=str, default=None, choices=['hse', 'dropout', 'augment'])
    parser.add_argument('--no_samples', type=int, default=50)

    args = parser.parse_args()
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    
    TEST_FN_PATH = '/media/HDD1/lavsen/dataset/camus-dataset/ImageSets/test_set.txt' 
    n_classes = 4
    test_fn = open(TEST_FN_PATH, 'r')
    test_fn_list = [line.split() for line in test_fn.readlines()]
    model = DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=True, freeze_bn=False, _print=True)

    if args.sampling_strategy is None:
        LOAD_PATH_MODEL = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar' 
        SAVE_PATH_MODEL_OUTPUT = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/pred_best_model/test_set/'
        
        save_output_images(LOAD_PATH_MODEL, model, test_loader, SAVE_PATH_MODEL_OUTPUT, test_fn_list, sampling_strategy= None)

    if args.sampling_strategy == 'hse':
        ALL_MODELS_PATH_HSE = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/all_models/camus_dataset/'
        model = torch.nn.DataParallel(model)  #Support for Parallel GPU 
        patch_replication_callback(model)
        model = model.cuda()
        starting_sample = 50
        base_save_path_output_hse = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/HSE/'
        if not (os.path.isdir(base_save_path_output_hse +'sample_0' )) :
            create_folders(starting_sample =starting_sample , path = base_save_path_output_hse, no_samples= args.no_samples)
        print ('Generating {} samples for HSE sampling strategy'.format(args.no_samples))
        for i in range(args.no_samples):
            LOAD_PATH_MODEL = ALL_MODELS_PATH_HSE + 'epoch_' + str(starting_epoch +i) 
            save_path_model_outputs_hse = base_save_path_output_hse + 'epoch_' + str(starting_epoch +i)  + '/'
            save_output_images(LOAD_PATH_MODEL,model, test_loader, save_path_model_outputs_hse, test_fn_list, sampling_strategy= args.sampling_strategy)

    if args.sampling_strategy == 'dropout':
        LOAD_PATH_MODEL = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar' 
        base_save_path_output_dropout = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/dropout/'
        print ('Generating {} samples for Dropout sampling strategy'.format(args.no_samples))
        if not (os.path.isdir(base_save_path_output_dropout +'sample_0' )) :
            create_folders(starting_sample =0 , path = base_save_path_output_dropout, no_samples= args.no_samples)
        for i in range(args.no_samples):
            save_path_model_outputs_dropout = base_save_path_output_dropout + 'sample_' + str(i)  + '/'
            save_output_images(LOAD_PATH_MODEL,model, test_loader, save_path_model_outputs_dropout, test_fn_list, sampling_strategy= args.sampling_strategy)


     if args.sampling_strategy == 'augment':
        LOAD_PATH_MODEL = '/home/lavsen/NAAMII/Projects/cardiac_seg/camus/pytorch-deeplab-xception/run/camus/deeplabv3plus-resnet-pretrained/model_best.pth.tar' 
        base_save_path_output_augment = '/media/HDD1/lavsen/all_research/2d_echo_uncertainty/outputs/sampling_strategy/augment/'
        print ('Generating {} samples for Dropout sampling strategy'.format(args.no_samples))
        if not (os.path.isdir(base_save_path_output_augment +'sample_0' )) :
            create_folders(starting_sample =0 , path = base_save_path_output_augment, no_samples= args.no_samples)
        