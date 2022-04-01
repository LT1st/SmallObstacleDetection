

import mypath
import os
import sys
import cv2
from tqdm import tqdm
import  numpy as np

root_path = mypath.Path.db_root_dir('small_obstacle')

folders=os.listdir(root_path)
tqdm_iter=tqdm(folders)
# 便利数据集文件夹
# ？？？这downtown_data_run到底是啥
# 根据数据label推测，不是本文数据集，可能是pretrain？
# 可能作者数据集和公开不一样，此处segmentation可能是label
for f in tqdm_iter:
	seg_path = os.path.join(root_path, f + '/label/')
	seg_files = os.listdir(seg_path)
	for files in seg_files:
		img = cv2.imread(os.path.join(seg_path, files))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# 获取标签
		all_colors = np.unique(img) # Dict: 107==Road 194 == of-road 241 == small obstacle
		# label 只有三种，超过了说明数据集错误

        print(all_colors)
