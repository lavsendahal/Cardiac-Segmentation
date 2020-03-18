class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/lavsen/NAAMII/dataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'camus_2ch_ed':
            return '/media/HDD1/lavsen/dataset/camus-dataset/'  # folder that contains dataset/.    
        elif dataset == 'camus':
            return '/media/HDD1/lavsen/dataset/camus-dataset/'  # folder that contains dataset/.  
        elif dataset == 'camus_lv':
            return '/media/HDD1/lavsen/dataset/camus-dataset/'  # folder that contains dataset/.     
        elif dataset == 'camus_lv_ed':
            return '/media/HDD1/lavsen/dataset/camus-dataset/'  # folder that contains dataset/.    
        elif dataset == 'camus_lv_es':
            return '/media/HDD1/lavsen/dataset/camus-dataset/'  # folder that contains dataset/.    
        elif dataset == 'isic_2017':
            return '/media/HDD1/lavsen/dataset/ISIC2017/'  # folder that contains dataset/.      
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'dynamic':
            return '/media/HDD1/lavsen/dataset/dynamic-dataset/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
