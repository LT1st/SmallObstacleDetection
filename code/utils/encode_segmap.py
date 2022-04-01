'''
本代码用于downtown_data_run数据集标签转化
/segmentation 转化为标准（0123）的标注数据 /label
为了和small obstacle 对齐
'''
import os
import sys
import cv2
from tqdm import tqdm
import  numpy as np

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

folders=os.listdir(root_path)
tqdm_iter=tqdm(folders)
# 遍历数据集文件夹
# ？？？这downtown_data_run到底是啥
# 根据数据label推测，不是本文数据集，可能是pretrain？
# 可能作者数据集和公开不一样，此处segmentation可能是label
for f in tqdm_iter:
	seg_path = os.path.join(root_path, f + '/segmentation/')
	seg_files = os.listdir(seg_path)
	for files in seg_files:
		img = cv2.imread(os.path.join(seg_path, files))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# 获取标签
		all_colors = np.unique(img) # Dict: 107==Road 194 == of-road 241 == small obstacle
		# label 只有三种，超过了说明数据集错误
		if len(all_colors)>3:
			raise RuntimeError("More than 3 classes in segmentation")

		img[img == 194] = 0
		img[img == 107] = 1
		img[img == 241] = 2
		write_path = os.path.join(root_path, f + '/labels/')
		if not os.path.exists(write_path):
			os.mkdir(write_path)
		cv2.imwrite(os.path.join(write_path,files), img)





