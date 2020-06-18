CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.007 --workers 4 --epochs 300 --batch-size 4 --gpu-ids 0 --checkname deeplabv3plus-resnet-pretrained --eval-interval 1 --dataset camus
