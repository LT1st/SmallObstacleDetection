CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 80 --batch-size 4 --gpu-ids 0,1 --mode train --dataset lnf 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --resume  /home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-image_input.pth --batch-size 2 --epoch 40 --mode train --ft --dataset lnf
