"""
计算数据集的 mean 和 std
"""
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
# import utils.helpers as HLP
import helpers as HLP
from mypath import Path
print('is running normalization metrics')

dataset = 'lnf'
train_imgs, train_disp, train_labels = HLP.get_ImagesAndLabels_mergenet(Path.db_root_dir(dataset))
test_imgs, test_disp, test_labels = HLP.get_ImagesAndLabels_mergenet(Path.db_root_dir(dataset),
                                          data_type='test')
train_loader = DataLoader(HLP.LNFGeneratorTorch(rgb_path=train_imgs,disparity_path=train_disp,
                                 mask_path=train_labels, flag = 'merge',
                                 split='train'), batch_size = 2, shuffle=True)
val_loader = DataLoader(HLP.LNFGeneratorTorch(rgb_path=test_imgs,disparity_path=test_disp,
                                 mask_path=test_labels, flag = 'merge',
                                 split='val'), batch_size=2, shuffle=True)
total_mean = []
total_std = []
for sample in tqdm(train_loader):
    image = sample['image']
    image = np.asarray(image)
    batch_mean = np.mean(image, axis=(0,2,3))
    total_mean.append(batch_mean/255)
    batch_std = np.std(image, axis=(0,2,3))
    total_std.append(batch_std/255)
for sample in tqdm(val_loader):
    image = sample['image']
    image = np.asarray(image)
    batch_mean = np.mean(image, axis=(0,2,3))
    total_mean.append(batch_mean/255)
    batch_std = np.std(image, axis=(0,2,3))
    total_std.append(batch_std/255)

mean = np.asarray(total_mean).mean(axis=0)
std = np.asarray(total_std).mean(axis=0)
print('mean: ', mean)
print('std: ', std)
