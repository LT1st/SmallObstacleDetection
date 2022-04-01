CUDA_VISIBLE_DEVICES=2,3 python train.py --resume \
/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-depth_input_uncertainty.pth \
--batch-size 2 --epoch 100 --mode train --ft --dataset lnf --gpu-ids 0,1 --depth \
--logsFlag naive_deeplab_confidence
