CUDA_VISIBLE_DEVICES=0 python train_pretrained.py --lr 0.007 --workers 4 --epochs 250 --batch-size 8 --gpu-ids 0 --checkname deeplabv3plus-pretrained-resnet_ed_and_es --eval-interval 1 --dataset camus_2ch_ed
