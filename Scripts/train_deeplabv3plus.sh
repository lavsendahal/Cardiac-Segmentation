CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.007 --workers 4 --epochs 500 --batch-size 4 --gpu-ids 0 --checkname deeplabv3plus-resnet --eval-interval 1 --dataset camus_2ch_ed
