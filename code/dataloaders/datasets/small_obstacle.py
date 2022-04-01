import os
import sys
sys.path.append('/home/ash/deeplab/')
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torchvision import transforms
from mypath import Path
from dataloaders import custom_transforms as tr
import numpy.ma as ma
from PIL import ImageFilter
from scipy.ndimage.filters import gaussian_filter

class SmallObs(data.Dataset):

	NUM_CLASSES = 3

	def __init__(self, args, image_paths, split='train'):
		'''
		:param args:
		:param image_paths: 图像路径
		:param split: 分割标志
		'''

		self.image_paths=image_paths
		self.split = split
		self.args = args
		self.root = self.image_paths
		self.trainSeq = os.listdir(os.path.join(self.root,self.split))
		'''
			这里有个问题，获取的路径不全，改了以后得适配下面的使用
			images_base  图像的文件夹
			input_files 提取出来的图像路径
			image_paths 图像文件夹
			
		'''
		self.images_base = []
		self.annotations_base = []
		# 拼接得到数据路径
		for seq in self.trainSeq:
			self.images_base_tmp = os.path.join(self.root, self.split, seq, 'image')
			self.images_base.append(self.images_base_tmp)
		# 最后是要 文件夹 还是图片
		#self.images_base = os.path.join(self.root,self.split,self.trainSeq,'image')
			# 数据集没这项啊
			self.annotations_base_tmp = os.path.join(self.root,self.split, seq, 'labels')
			self.annotations_base.append(self.annotations_base_tmp)

		self.class_names = ['off_road','on_road','small_obstacle']

		self.input_files = []
		for image_base in self.images_base:
			for ii in os.listdir(image_base):
				# 'D:\\00code\\small_obstacle_discovery-master\\dataloaders\\datasets\\0000000080.png'
				# thispath = os.path.abspath(ii)
				if ii != '.DS_Store':
					thispath = os.path.join(image_base, ii)
					self.input_files.append(thispath)

		self.input_files = sorted(self.input_files)

		# 裁切结束尺寸
		finalHeight = 512
		# 计算图像裁切尺寸
		self.imgHeight = np.asarray(Image.open(self.input_files[0])).shape[0]
		self.imgLength = np.asarray(Image.open(self.input_files[0])).shape[1]
		self.imgChannel = np.asarray(Image.open(self.input_files[0])).shape[2]
		assert finalHeight<self.imgHeight, '输入图太小'

		# x1,x2,y1,y2,c
		self.img = [(self.imgHeight-finalHeight)//2, (self.imgHeight-finalHeight)//2+finalHeight,
					self.imgLength, self.imgLength, self.imgChannel]

		# 固定排序顺序的img list
		# self.input_files = sorted(os.listdir(self.images_base))
		if len(self.input_files)==0:
			raise Exception("No files found in directory: {}".format(self.images_base))
		else:
			print("Found %d images" % (len(self.input_files)))


	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		'''
		根据index返回具体图像
		:param index: 图像索引，不能超过 __len__(self)
		:return: 具体图像
		'''

		# 图片的文件目录是F:\Small_Obstacle_Dataset\train\file_1\image\xxx.jpg
		'''self的参数说明
		image_paths 所有图像的路径

		'''
		# 图像路径
		input_path = self.input_files[index]
		temp=input_path.split('image')
		# 标签路径
		target_path = temp[0] + 'labels' + temp[1]

		# 输入图象是RGBA A通道不为空
		"""
		19337,  22855,  32762,  48936,  61543,  63971,  71909,  71794,
        66344,  60794,  54561,  48849,  43106,  39278,  35932,  33432,
        31244,  29016,  27962,  26472,  25666,  26260,  25795,  25817,
        25904,  25461,  25170,  25842,  25074,  25760,  25249,  23705,
        22903,  22428,  21600,  20844,  19855,  18662,  17787,  16803,
        16421,  15907,  15047,  14733,  14088,  13655,  13080,  12510,
        11822,  11590,  11147,  11227,  10822,  10837,  10244,   9831,
         9440,   8997,   8591,   8352,   8226,   7851,   7686,   7252,
         7032,   6956,   6740,   6745,   6276,   6188,   5930,   5683,
         5520,   5176,   5082,   4919,   4799,   4670,   4329,   4346,
         4147,   3954,   3784,   3662,   3584,   3432,   3353,   3182,
         3151,   2951,   2810,   2715,   2604,   2605,   2493,   2457,
         2521,   2530,   2489,   2392,   2422,   2439,   2329,   2399,
         2483,   2380,   2426,   2617,   2821,   2960,   3061,   3100,
         3395,   3648,   4039,   4662,   5248,   5641,   6573,   7613,
         9181,  10749,  12287,  14140,  15783,  16550,  17661,  18408,
        19789,  20520,  21849,  23072,  23663,  24591,  24326,  24802,
        24461,  25026,  25314,  24365,  23434,  22880,  21864,  20621,
        18897,  16866,  15397,  14086,  12548,  11266,  10058,   9175,
         7937,   6863,   5947,   5264,   4609,   4017,   3701,   3284,
         3115,   2977,   2639,   2479,   2310,   2146,   2132,   2045,
         1948,   1830,   1688,   1786,   1779,   1672,   1592,   1671,
         1640,   1537,   1649,   1595,   1698,   1569,   1585,   1594,
         1677,   1647,   1663,   1638,   1604,   1711,   1638,   1656,
         1682,   1671,   1608,   1649,   1475,   1531,   1490,   1467,
         1383,   1320,   1206,   1181,   1286,   1289,   1292,   1183,
         1096,   1166,   1106,   1044,   1060,   1054,   1045,   1016,
          889,    932,    903,    890,   1002,   1033,   1097,   1023,
         1085,    989,    996,   1016,   1113,   1052,   1126,   1080,
         1250,   1380,   1353,   1439,   1458,   1714,   1519,   1702,
         1675,   1619,   1708,   1826,   2055,   2230,   2678,   3245,
         4426,   5836,   6335,   8083,  14082,  32928,  13391, 980280],
      dtype=int64)

		"""
		# 裁切中间的512输入
		'''
		警告 1280 x 720
		那不是512怎么办？？？？？？？？？
		'''
		'''
		报错
		720高度的图片无法正确裁切
		'''
		# import cv2
		# from matplotlib import pyplot as plt
		# plt.imshow(np.asarray(Image.open(input_path))[:,:,3])
		# show = Image.open(input_path).numpy()
		# cv2.imshow(image)

		_img = np.asarray(Image.open(input_path))[self.img[0]:self.img[1], :, :3]
		_target = np.asarray(Image.open(target_path))[self.img[0]:self.img[1], :]
		_img=Image.fromarray(_img)
		_target=Image.fromarray(_target)
		# 对应关系字典
		sample={'image':_img,'label':_target}

		if self.split == 'train':
			return self.transform_tr(sample)

		elif self.split == 'val':
			return self.transform_val(sample)

		elif self.split == 'test':
			return self.transform_ts(sample)


	def transform_tr(self,sample):
		'''
		返回随机水平反转、随机裁切、标准化处理的训练数据
		:param sample: sample={'image':_img,'label':_target}，为PIL格式
		:return: tensor格式的图像
		'''
		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),# 数据增强，随机水平反转
			tr.RandomCrop(crop_size=(512,512)),# 随机裁切，但是源代码限制512*512，参考警告
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),# 归一
			tr.ToTensor()# 转tensor
			])
		return composed_transforms(sample)

	def transform_val(self,sample):
		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),
			tr.RandomCrop(crop_size=(512,512)),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])

		return composed_transforms(sample)

	def transform_ts(self,sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])

		return composed_transforms(sample)


class SmallObs_RGBD(data.Dataset):
	NUM_CLASSES = 3

	def __init__(self, args, image_paths, split='train'):
		'''
		:param args: 启动参数
		:param image_paths: 数据集总路径
		:param split: 分割标志
		'''

		print('Using SmallObs_RGBD dataloader')
		self.image_paths = image_paths
		self.split = split
		self.args = args
		self.root = self.image_paths
		self.trainSeq = os.listdir(os.path.join(self.root, self.split))
		'''
			这里有个问题，获取的路径不全，改了以后得适配下面的使用
			images_base  图像的文件夹
			input_files 提取出来的图像路径
			image_paths 图像文件夹
		'''
		self.images_base = []
		self.annotations_base = []
		self.depths_base = []
		# 拼接得到数据路径
		for seq in self.trainSeq:
			self.images_base_tmp = os.path.join(self.root, self.split, seq, 'image')
			self.images_base.append(self.images_base_tmp)
			# 最后是要 文件夹 还是图片
			# self.images_base = os.path.join(self.root,self.split,self.trainSeq,'image')
			# 数据集没这项啊 改一下
			self.annotations_base_tmp = os.path.join(self.root, self.split, seq, 'labels')
			self.annotations_base.append(self.annotations_base_tmp)
			# depth
			self.depths_base_tmp = os.path.join(self.root, self.split, seq, 'depth')
			self.depths_base.append(self.depths_base_tmp)

		self.class_names = ['off_road', 'on_road', 'small_obstacle']

		self.input_files = []
		for image_base in self.images_base:
			for ii in os.listdir(image_base):
				# 'D:\\00code\\small_obstacle_discovery-master\\dataloaders\\datasets\\0000000080.png'
				# thispath = os.path.abspath(ii)
				if ii != '.DS_Store':
					thispath = os.path.join(image_base, ii)
					self.input_files.append(thispath)

		self.input_files = sorted(self.input_files)

		# 裁切结束尺寸
		finalHeight = 512
		# 计算图像裁切尺寸
		self.imgHeight = np.asarray(Image.open(self.input_files[0])).shape[0]
		self.imgLength = np.asarray(Image.open(self.input_files[0])).shape[1]
		self.imgChannel = np.asarray(Image.open(self.input_files[0])).shape[2]
		assert finalHeight < self.imgHeight, '输入图太小'

		# x1,x2,y1,y2,c
		self.img = [(self.imgHeight - finalHeight) // 2, (self.imgHeight - finalHeight) // 2 + finalHeight,
					self.imgLength, self.imgLength, self.imgChannel]

		# 固定排序顺序的img list
		# self.input_files = sorted(os.listdir(self.images_base))
		if len(self.input_files) == 0:
			raise Exception("No files found in directory: {}".format(self.images_base))
		else:
			print("Found %d images" % (len(self.input_files)))

	def __len__(self):
		return len(self.input_files)

	def __getitem__(self, index):
		'''
		根据index返回具体图像
		:param index: 图像索引，不能超过 __len__(self)
		:return: 具体图像
		'''

		# 图片的文件目录是F:\Small_Obstacle_Dataset\train\file_1\image\xxx.jpg
		'''self的参数说明
		image_paths 所有图像的路径

		'''
		# 图像路径
		input_path = self.input_files[index]
		temp = input_path.split('image')
		# 标签路径
		target_path = temp[0] + 'labels' + temp[1]
		depth_path = temp[0] + 'depth' + temp[1]

		# 输入图象是RGBA A通道不为空
		"""
		19337,  22855,  32762,  48936,  61543,  63971,  71909,  71794,
        66344,  60794,  54561,  48849,  43106,  39278,  35932,  33432,
        31244,  29016,  27962,  26472,  25666,  26260,  25795,  25817,
        25904,  25461,  25170,  25842,  25074,  25760,  25249,  23705,
        22903,  22428,  21600,  20844,  19855,  18662,  17787,  16803,
        16421,  15907,  15047,  14733,  14088,  13655,  13080,  12510,
        11822,  11590,  11147,  11227,  10822,  10837,  10244,   9831,
         9440,   8997,   8591,   8352,   8226,   7851,   7686,   7252,
         7032,   6956,   6740,   6745,   6276,   6188,   5930,   5683,
         5520,   5176,   5082,   4919,   4799,   4670,   4329,   4346,
         4147,   3954,   3784,   3662,   3584,   3432,   3353,   3182,
         3151,   2951,   2810,   2715,   2604,   2605,   2493,   2457,
         2521,   2530,   2489,   2392,   2422,   2439,   2329,   2399,
         2483,   2380,   2426,   2617,   2821,   2960,   3061,   3100,
         3395,   3648,   4039,   4662,   5248,   5641,   6573,   7613,
         9181,  10749,  12287,  14140,  15783,  16550,  17661,  18408,
        19789,  20520,  21849,  23072,  23663,  24591,  24326,  24802,
        24461,  25026,  25314,  24365,  23434,  22880,  21864,  20621,
        18897,  16866,  15397,  14086,  12548,  11266,  10058,   9175,
         7937,   6863,   5947,   5264,   4609,   4017,   3701,   3284,
         3115,   2977,   2639,   2479,   2310,   2146,   2132,   2045,
         1948,   1830,   1688,   1786,   1779,   1672,   1592,   1671,
         1640,   1537,   1649,   1595,   1698,   1569,   1585,   1594,
         1677,   1647,   1663,   1638,   1604,   1711,   1638,   1656,
         1682,   1671,   1608,   1649,   1475,   1531,   1490,   1467,
         1383,   1320,   1206,   1181,   1286,   1289,   1292,   1183,
         1096,   1166,   1106,   1044,   1060,   1054,   1045,   1016,
          889,    932,    903,    890,   1002,   1033,   1097,   1023,
         1085,    989,    996,   1016,   1113,   1052,   1126,   1080,
         1250,   1380,   1353,   1439,   1458,   1714,   1519,   1702,
         1675,   1619,   1708,   1826,   2055,   2230,   2678,   3245,
         4426,   5836,   6335,   8083,  14082,  32928,  13391, 980280],
      dtype=int64)

		"""
		# 裁切中间的512输入
		'''
		警告 1280 x 720
		那不是512怎么办？？？？？？？？？
		'''
		'''
		报错
		720高度的图片无法正确裁切
		'''
		_img = np.asarray(Image.open(input_path))[self.img[0]:self.img[1], :, :3]
		_target = np.asarray(Image.open(target_path))[self.img[0]:self.img[1], :]
		_depth = np.asarray(Image.open(depth_path))[self.img[0]:self.img[1], :]

		# mask
		mask = _target > 1
		# print(_depth[mask])

		# 处理标签 原来的》1都是障碍物 这里得变成2是障碍物
		# %timeit 测试 利用mask慢一点
		#_target = ma.masked_array(_target, mask=mask,fill_value=2)
		tst = _target.copy()
		tst[mask] = 2
		_target = tst

		# 点乘，利用mask
		# 需要加入大概率的随机擦除，不然过拟合严重（相当于是label泄露）
		_depth = np.multiply(_depth, mask)

		# int32转换uint8
		# max = 65535，是16bit
		_depth = _depth / 256
		_depth = np.rint(_depth)
		_depth = _depth.astype(np.uint8)

		# 增加一个维度，用于拼接
		_depth = _depth.reshape((_depth.shape[0], _depth.shape[1], 1))# 也可以
		# _depth = np.expand_dims(_depth, axis=2)

		# 注意 这里的图像格式问题
		# _img = Image.fromarray(_img)
		# _target = Image.fromarray(_target)


		# 对深度图执行高斯模糊
		# _depth = Image.fromarray(_depth)
		# _depth = _depth.filter(ImageFilter.GaussianBlur(radius=10))
		# _depth = np.asarray(_depth)

		_depth = gaussian_filter(_depth, sigma=7)

		# 拼接
		'''
		报错
		TypeError: concatenate() got multiple values for argument 'axis'
		拼接[_img, _depth]
		'''
		'''
		报错
		拼接无效 
		depth为int32，需要转换为uint8
		'''
		_img = np.concatenate((_img, _depth), axis=2)
		# _img = Image.fromarray(_img)

		# 对应关系字典
		# 注意 这里的图像格式问题
		'''
		报错
		raise TypeError("Cannot handle this data type: %s, %s" % typekey) from e
		TypeError: Cannot handle this data type: (1, 1, 4), <i4
		解决
		图像数据格式问题
		'''
		sample = {'image': Image.fromarray(_img), 'label': Image.fromarray(_target)}

		if self.split == 'train':
			return self.transform_tr(sample)

		elif self.split == 'val':
			return self.transform_val(sample)

		elif self.split == 'test':
			return self.transform_ts(sample)

	def transform_tr(self, sample):
		'''
		返回随机水平反转、随机裁切、标准化处理的训练数据
		:param sample: sample={'image':_img,'label':_target}，为PIL格式
		:return: tensor格式的图像
		'''
		'''
		TODO 警告
		这里的depth维度未进行完整统计
		'''
		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),  # 数据增强，随机水平反转
			tr.RandomCrop(crop_size=(512, 512)),  # 随机裁切，但是源代码限制512*512，参考警告

			tr.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.512)),  # 归一
			tr.ToTensor()  # 转tensor
		])
		return composed_transforms(sample)

	def transform_val(self, sample):
		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),
			tr.RandomCrop(crop_size=(512, 512)),
			tr.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.512)),
			tr.ToTensor()])

		return composed_transforms(sample)

	def transform_ts(self, sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.512)),
			tr.ToTensor()])

		return composed_transforms(sample)


if __name__ == '__main__':
	from dataloaders.utils import decode_segmap
	from torch.utils.data import DataLoader
	import matplotlib.pyplot as plt
	import argparse

	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	args.base_size = 512
	args.crop_size = 512
	path = Path.db_root_dir('small_obstacle')
	cityscapes_train = SmallObs_RGBD(args=args, image_paths=path, split='train')
	'''
	报错 已修改
	由于文件路径不符合格式，迭代器错误
	'''
	trainloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)
	# images,labels=next(iter(trainloader))
	pairs = next(iter(trainloader))
	images, labels = pairs['image'],pairs['label']
	print(np.asarray(images).shape,np.asarray(labels).shape,type(labels))
