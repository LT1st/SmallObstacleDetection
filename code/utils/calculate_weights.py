import os
from tqdm import tqdm
import numpy as np
from mypath import Path
'''
计算各label下的覆盖率， 用于计算后续的LOSS
class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
'''
def calculate_weigths_labels(dataset, dataloader, num_classes):
	'''
	function:按照标签准确率来确定权值
	:param dataset:
	:param dataloader: 传入的数据要求标签做过归一化（n< num_classes）
	:param num_classes:
	:return:
	'''
	# Create an instance from the data loader
	z = np.zeros((num_classes,))
	# 比如 array([0., 0., 0.])
	# Initialize tqdm
	tqdm_batch = tqdm(dataloader)
	print('Calculating classes weights')
	for sample in tqdm_batch:
		y = sample['label']
		y = y.detach().cpu().numpy()
		# 获取标注
		'''
		报错
		RuntimeError: CUDA error: device-side assert triggered
		C:/cb/pytorch_1000000000000/work/aten/src/THCUNN/SpatialClassNLLCriterion.cu:106: block: [3,0,0], thread: [651,0,0] Assertion `t >= 0 && t < n_classes` failed.
		'''
		mask = (y >= 0) & (y < num_classes)
		labels = y[mask].astype(np.uint8)
		# np.bincount()巧妙计算覆盖量 每个label有多少像素
		count_l = np.bincount(labels, minlength=num_classes)
		z += count_l
	tqdm_batch.close()
	# 总覆盖量
	total_frequency = np.sum(z)
	class_weights = []
	# 遍历每个dataloader送来的覆盖量
	for frequency in z:
		# 计算得到的是 单张图片准确像素数量 在 总准确像素数量 中占的比重
		class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
		class_weights.append(class_weight)#list
	ret = np.array(class_weights)

	# 在数据集路径下写结果 XXX_classes_weights.npy
	classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
	open(classes_weights_path,"wb").close()
	np.save(classes_weights_path, ret)
	return ret


def calculate_weights_batch(sample,num_classes):
	'''
	计算
	:param sample: 一个batch的 所有 图像和对应标签
	:param num_classes: 类别数
	:return:
	'''
	z = np.zeros((num_classes,))
	y = sample['label']
	y = y.detach().cpu().numpy()
	mask = (y >= 0) & (y < num_classes)
	labels = y[mask].astype(np.uint8)
	# 每个label有多少像素
	count_l = np.bincount(labels, minlength=num_classes)
	# ？这个逻辑从上面直接抄下来了？ z是啥debug看一下
	# 不影响
	z += count_l
	total_frequency = np.sum(z)
	class_weights = []
	for frequency in z:
		class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
		class_weights.append(class_weight)
	# 每个标签对应一个类别权值 表示此batch内标签的数量分布
	ret = np.array(class_weights)
	return ret