"""
用于探索数据集
"""
import os
import sys
import cv2
from tqdm import tqdm
import  numpy as np
from collections import Counter
# root_path= os.getcwd()+'/downtown_data_run/'
# root_path = os.path.join(os.getcwd(),'downtown_data_run/')
# 将项目主目录加入search space
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), '..'))
'''
报错 : import mypath
linux和pycharm不同，没有系统设置的路径，需要自己找路径
'''
import mypath

root_path = mypath.Path.db_root_dir('small_obstacle')
root_path = os.path.join(root_path,'train')
folders=os.listdir(root_path)
tqdm_iter=tqdm(folders)
# 便利数据集文件夹
# ？？？这downtown_data_run到底是啥
# 根据数据label推测，不是本文数据集，可能是pretrain？
# 可能作者数据集和公开不一样，此处segmentation可能是label
all_colors = np.ndarray([0],dtype='int16')
for f in tqdm_iter:
	seg_path = os.path.join(root_path, f + '/labels/')
	seg_files = os.listdir(seg_path)

	for files in seg_files:
		img = cv2.imread(os.path.join(seg_path, files))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# 获取标签
		all_colors = np.hstack((all_colors,np.unique(img)))

		#all_colors.append(np.unique(img)) # Dict: 107==Road 194 == of-road 241 == small obstacle
		# label 只有三种，超过了说明数据集错误
		#print(all_colors)

# Counter({0: 1932, 38: 1932, 128: 914, 75: 653, 57: 562, 90: 287, 113: 192, 53: 183, 15: 41})
print(all_colors,Counter(all_colors))
arr = np.bincount(all_colors)
Counter(all_colors)





