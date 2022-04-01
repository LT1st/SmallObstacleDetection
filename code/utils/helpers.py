import colorsys
from torchvision import transforms
'''
用于数据加载
'''
import pdb
from termcolor import colored
from matplotlib.pyplot import imsave
from matplotlib.colors import LinearSegmentedColormap
import argparse
from PIL import Image
from multiprocessing import Pool
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from dataloaders import custom_transforms as tr



# calculate weighted loss
def calculate_weights_batch(z):
	total_frequency = np.sum(z)
	class_weights = []
	for frequency in z:
		class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
		class_weights.append(class_weight)
	ret = np.array(class_weights)
	return ret

# For stripe net
# manually creating stripes out of an image without using reshape
def strip_image_new(image=None, path=None, stripe_size=32, flag='image', batch_size=1):
    '''
        生成条形网
        :param image:
        :param path:
        :param stripe_size:
        :param flag: 传进来的图像类型
        :param batch_size:
        :return:
        '''
    image_arr = []
    if path is not None:
        image = np.asarray(Image.open(path, 'r'))
    if flag == 'image' and path is not None:
        im_cropped = image[281:793, 128:1920, :]
        for i in range(0, im_cropped.shape[1], stripe_size):
            image_arr.append(im_cropped[:, i:i+stripe_size, :])
        result = np.asarray(image_arr)
        return result
    elif flag == 'image' and path is None:
        for k in range(batch_size):
            for i in range(0, image.shape[2], stripe_size):
                image_arr.append(image[k, :,  i:i+stripe_size, :])
        result = np.reshape(np.asarray(image_arr), (batch_size*56, 512, 32, 3))
        return result
    elif flag == 'disparity' and path is not None:
        im_cropped = image[281:793, 128:1920]
        im_cropped = depth_preprocessing(im_cropped)
        for i in range(0, im_cropped.shape[1], stripe_size):
            image_arr.append(im_cropped[:, i:i+stripe_size])

        result = np.asarray(image_arr)
        result = np.reshape(result, (56, 512, 32, 1))
        return result
    elif flag == 'disparity' and path is None:
        for k in range(batch_size):
            for i in range(0, image.shape[2], stripe_size):
                image_arr.append(image[k, :, i:i+stripe_size, :])
        result = np.reshape(np.asarray(image_arr), (batch_size*56, 512, 32, 1))
        return result
    elif flag == 'mask':
        im_cropped = image.copy()
        im_cropped = im_cropped[281:793, 128:1920]
        im_cropped[im_cropped == 255] = 0
        for i in range(0, im_cropped.shape[1], stripe_size):
            image_arr.append(im_cropped[:, i:i+stripe_size])

        return np.asarray(image_arr)


# pre-processing depth images, changing it from a 16 bit value to an 8 bit
# value
def depth_preprocessing(img):
    '''
    深度图16bit转8bit
    :param img:
    :return:
    '''
    img = img/256
    # 四舍五入
    img = np.rint(img)
    img = img.astype(np.uint8)
    return img

# 从条形图恢复
def unstripe_new(image, flag='image', stripe_size=32, batch_size=1,
                 num_stripes=56):
    '''
    从条形图恢复
    :param image:
    :param flag:
    :param stripe_size:
    :param batch_size:
    :param num_stripes:
    :return:
    '''
    if flag == 'image':
        image_arr = np.zeros((512, 1792, 3))
        for i in range(image.shape[0]):
            image_arr[:, i*stripe_size:(i+1)*stripe_size, :] = image[i]
        return image_arr
    elif flag == 'mask':
        image_arr = np.zeros((512, 1792))
        for i in range(image.shape[0]):
            image_arr[:, i*stripe_size:(i+1)*stripe_size] = image[i]
        return image_arr
    elif flag == 'softmax':
        image_arr = np.zeros((batch_size, 3, 512, 1792))
        n_images = int(image.shape[0]/56)
        for k in range(n_images):
            for i in range(num_stripes):
                image_arr[k, :, :, i*stripe_size:(i+1)*stripe_size] = image[i, :, :, :]
        return image_arr

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    image = image.copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 2,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

# batches stripe paths 
def get_ImagesAndLabels_from_dir(path, data_type='train', num_stripes=56, number_of_samples=None):
    """
    根据给定数据集路径， 获取CityPath类型数据集的图像路径
    :param path:
    :param data_type:
    :param num_stripes: 切条数量
    :param number_of_samples: 获取图像数量
    :return: images, disparities, labels
    """
    images = []
    labels = []
    disparities = []
    image_path = path+'leftImg8bit/'+data_type+'/'
    label_path = path+'gtCoarse/'+data_type+'/'
    disparity_path = path+'disparity/'+data_type+'/'
    list_of_paths = sorted(os.listdir(image_path))
    for a in list_of_paths:
        if a != '.DS_Store':
            temp = os.path.join(image_path, a)
            list_of_paths = os.listdir(temp)
            for img in list_of_paths:
                for i in range(num_stripes):
                    images.append(os.path.join(temp, img)+'.m'+str(i)+'n')
                label = img.split('_leftImg8bit.png')[0]+'_gtCoarse_labelTrainIds.png'
                label = os.path.join(label_path, a, label)
                for i in range(num_stripes):
                    labels.append(label+'.m'+str(i)+'n')
                disparity = img.split('_leftImg8bit.png')[0]+'_disparity.png'
                disparity = os.path.join(disparity_path, a, disparity)
                for i in range(num_stripes):
                    disparities.append(disparity+'.m'+str(i)+'n')

    # 打包，保证shuffle对齐
    result = list(zip(images, disparities, labels))
    random.shuffle(result)
    # 利用* 自动解包
    images, disparities, labels = zip(*result)
    if number_of_samples is not None:
        return images[:num_stripes*number_of_samples], disparities[:num_stripes*number_of_samples], labels[:num_stripes * number_of_samples]
    else:
        return images, disparities, labels

# creates lists of image and label paths
def get_ImagesAndLabels_contextnet(path, data_type='train', num_samples=None):
    '''
    用的一直是这个, 文中context指的是： RGB+lidar。这里只有RGB
    :param path: 文件夹总路径
    :param data_type: 用训练集还是测试集 train test 二选一
    :param num_samples: 总数（空
    :return: 经过shuffle的 images, labels 的 list（zip拼接）
    '''
    images= []
    labels = []
    image_path = path+'leftImg8bit/'+data_type+'/'
    label_path = path+'gtCoarse/'+data_type+'/'
    for a in sorted(os.listdir(image_path)):
        if a != '.DS_Store':
            temp = os.path.join(image_path, a)
            for img in os.listdir(temp):
                images.append(os.path.join(temp, img))
                label = img.split('_leftImg8bit.png')[0]+'_gtCoarse_labelTrainIds.png'
                label = os.path.join(label_path, a, label)
                labels.append(label)

    result = list(zip(images, labels))
    random.shuffle(result)
    images, labels = zip(*result)
    if num_samples is not None:
        return images[:num_samples], labels[:num_samples]
    else:
        return images, labels

# image paths for mergenet training
def get_ImagesAndLabels_mergenet(path, data_type='train', num_samples=None):
    '''
    用于LF数据集 的RGB+深度融合 获取深度图和RGB的文件路劲
    :param path: 数据集路径
    :param data_type: 数据 划分选择
    :param num_samples: 提取图像数量
    :return: 深度图和RGB的文件路劲 经过shuffle的 images, labels 的 list（zip拼接）
    '''
    images= []
    labels = []
    disparity = []
    image_path = path+'leftImg8bit/'+data_type+'/'
    label_path = path+'gtCoarse/'+data_type+'/'
    disparity_path = path+'disparity/'+data_type+'/'
    for a in sorted(os.listdir(image_path)):
        if a != '.DS_Store':
            # 场景文件夹
            temp = os.path.join(image_path, a)
            # 遍历每个场景文件夹
            for img in os.listdir(temp):

                images.append(os.path.join(temp, img))
                label = img.split('_leftImg8bit.png')[0]+'_gtCoarse_labelTrainIds.png'
                depth = img.split('_leftImg8bit.png')[0]+'_disparity.png'
                depth = os.path.join(disparity_path, a, depth)
                label = os.path.join(label_path, a, label)
                disparity.append(depth)
                labels.append(label)

    result = list(zip(images, disparity, labels))
    # 注意 shuffle
    random.shuffle(result)
    images, disparity, labels = zip(*result)
    if num_samples is not None:
        return images[:num_samples], disparity[:num_samples], labels[:num_samples]
    else:
        return images, disparity, labels

def get_ImagesAndLabels_mergenet_small_obstacle_dataset(path, data_type='train', num_samples=None, shuf=True):
    '''
    用于small_obstacle_dataset数据集 的 RGB+深度融合 获取深度图和RGB的文件路
    :param path: 数据集路径
    :param data_type: 数据 划分选择
    :param num_samples: 提取图像数量
    :return: 深度图和RGB的文件路劲 经过shuffle的 images, labels 的 list（zip拼接）
    '''
    images= []
    labels = []
    disparity = []
    image_path = path+'leftImg8bit/'+data_type+'/'
    label_path = path+'gtCoarse/'+data_type+'/'
    disparity_path = path+'disparity/'+data_type+'/'
    for a in sorted(os.listdir(image_path)):
        if a != '.DS_Store':
            # 场景文件夹
            temp = os.path.join(image_path, a)
            # 遍历每个场景文件夹
            for img in os.listdir(temp):

                images.append(os.path.join(temp, img))
                label = img.split('_leftImg8bit.png')[0]+'_gtCoarse_labelTrainIds.png'
                depth = img.split('_leftImg8bit.png')[0]+'_disparity.png'
                depth = os.path.join(disparity_path, a, depth)
                label = os.path.join(label_path, a, label)
                disparity.append(depth)
                labels.append(label)

    result = list(zip(images, disparity, labels))
    # 注意 shuffle
    if shuf:
        random.shuffle(result)
    images, disparity, labels = zip(*result)
    if num_samples is not None:
        return images[:num_samples], disparity[:num_samples], labels[:num_samples]
    else:
        return images, disparity, labels

def generate_additional_stripes(images, disparities, labels, path, width=32, num_stripes=56, stride=9, data_type='train', step_size=32):
    images = list(images)
    disparities = list(disparities)
    labels = list(labels)
    image_path = path+'leftImg8bit/'+data_type+'/'
    label_path = path+'gtCoarse/'+data_type+'/'
    disparity_path = path+'disparity/'+data_type+'/'
    for a in sorted(os.listdir(image_path)):
        if a != '.DS_Store':
            temp = os.path.join(image_path, a)
            for img in os.listdir(temp):
                label = img.split('_leftImg8bit.png')[0]+'_gtCoarse_labelTrainIds.png'
                label = os.path.join(label_path, a, label)
                im = np.asarray(Image.open(label, 'r'))
                im_cropped = im[281:793, 128:1920]
                start = width
                end = im_cropped.shape[1]-width
                for i in range(start, end, step_size):
                    temp_im = im_cropped[:, i-width:i]
                    b = np.sum(temp_im==2)
                    if b > 0:
                        for k in range(i-width, i, stride):
                            labels.append(label+'.n'+str(k)+'r')
                            images.append(os.path.join(temp, img)+'.n'+str(k)+'r')
                            disparity = img.split('_leftImg8bit.png')[0]+'_disparity.png'
                            disparity = os.path.join(disparity_path, a, disparity)
                            disparities.append(disparity+'.n'+str(k)+'r')

    result = list(zip(images, disparities, labels))
    random.shuffle(result)
    images, disparities, labels = zip(*result)
    return images, disparities, labels

# dataset loader for torch
class LNFGeneratorTorch(Dataset):
    def __init__(self, rgb_path, disparity_path=None, mask_path=None,
                 flag='stripe', split='train', batch_size=32, pool_size=5, stripe_size=32, **kwargs):
        '''
        # 0条纹网络'stripe'  1语义网络'context' 2合并网络'merge'
        Initializing paths for the rgb/disparity features and mask labels
        if flag = 0, the data generator is in stripenet training mode
        if flag = 1, the data generator is in contextnet training mode
        '''
        self._x_rgb = rgb_path
        self._x_dis = disparity_path
        self._y_mask = mask_path
        self._batch_size = batch_size
        self.flag = flag
        self.split = split
        self._pool_size = pool_size
        if self.flag == 'stripe':
            self._rgb_pool = Pool(self._pool_size)
            self._disparity_pool = Pool(self._pool_size)
            self._label_pool = Pool(self._pool_size)
        elif self.flag == 'context':
            self._context_rgb_pool = Pool(self._pool_size)
            self._context_label_pool = Pool(self._pool_size)
        elif self.flag == 'merge':
            self._mergenet_rgb_pool = Pool(self._pool_size)
            self._mergenet_disp_pool = Pool(self._pool_size)
            self._mergenet_label_pool = Pool(self._pool_size)
        self._stripe_size = stripe_size

    # overloads []
    def __getitem__(self, index):
        '''
        generates batches for training
        '''
        if self.flag == 'stripe':
            if index == (self.__len__() - 1):
                batch_x_rgb = self._x_rgb[index * self._batch_size:]
                batch_x_dis = self._x_dis[index * self._batch_size:]
                batch_y_mask = self._y_mask[index * self._batch_size:]
            else:
                batch_x_rgb = self._x_rgb[index * self._batch_size:(index+1) * self._batch_size]
                batch_x_dis = self._x_dis[index * self._batch_size:(index+1) * self._batch_size]
                batch_y_mask = self._y_mask[index * self._batch_size:(index+1) * self._batch_size]

            # # creation of tensors
            # X_rgb, X_dis = self._create_feature_tensor(batch_x_rgb, batch_x_dis)
            # Y_mask = self._create_label_tensor(batch_y_mask)
            # return ([X_rgb, X_dis], Y_mask) 

            X_rgb, X_dis = self._stripenet_feature_tensor_pool(batch_x_rgb, batch_x_dis)
            Y_mask = self._stripenet_label_tensor_pool(batch_y_mask)
            return ([np.asarray(X_rgb), np.asarray(X_dis)],
                    np.asarray(Y_mask))



        elif self.flag == 'context':

            X_rgb = LNFGeneratorTorch._context_func_rgb(self._x_rgb[index])
            Y_mask = LNFGeneratorTorch._context_func_labels(self._y_mask[index])
            sample = {'image':Image.fromarray(np.asarray(X_rgb)),
                      'label':Image.fromarray(np.asarray(Y_mask))}
            # 几个操作一样
            if self.split == 'train':
                    return self.transform_tr(sample)

            elif self.split == 'val':
                    return self.transform_val(sample)

            elif self.split == 'test':
                    return self.transform_ts(sample)



        elif self.flag == 'merge':
            # GRB
            X_rgb = LNFGeneratorTorch._mergenet_func_rgb(self._x_rgb[index])
            # 深度
            X_disp = LNFGeneratorTorch._mergenet_func_disparity(self._x_dis[index])
            # 标签
            Y_mask = LNFGeneratorTorch._mergenet_func_labels(self._y_mask[index])
            # 拼接第四维
            X_ft = np.concatenate((np.asarray(X_rgb), np.asarray(X_disp)), axis=2)
            sample = {'image':Image.fromarray(X_ft),
                      'label':Image.fromarray(np.asarray(Y_mask))}
            if self.split == 'train':
                return self.transform_tr_depth(sample)
            elif self.split == 'val':
                return self.transform_val_depth(sample)
            elif self.split == 'test':
                return self.transform_ts_depth(sample)

    # overloads len()
    def __len__(self):
        return len(self._x_rgb)


    def _mergenet_feature_tensor_pool(self, rgb_paths, dis_paths):
        something_rgb = self._mergenet_rgb_pool.map(LNFGeneratorTorch._mergenet_func_rgb,
                                    rgb_paths)
        something_disparity = self._mergenet_disp_pool.map(LNFGeneratorTorch._mergenet_func_disparity,
                                         dis_paths)
        return (something_rgb, something_disparity)

    def _mergenet_label_tensor_pool(self, paths):
        labels = self._mergenet_label_pool.map(LNFGeneratorTorch._mergenet_func_labels,
                                     paths)
        return labels

    def _stripenet_feature_tensor_pool(self, rgb_paths, dis_paths):
        something_rgb = self._rgb_pool.map(LNFGeneratorTorch._func_rgb, rgb_paths)
        something_disparity = self._disparity_pool.map(LNFGeneratorTorch._func_disparity,
                                                       dis_paths)
        return (something_rgb, something_disparity)

    def _stripenet_label_tensor_pool(self, paths):
        labels = self._label_pool.map(LNFGeneratorTorch._func_labels, paths)
        return labels

    def _contextnet_feature_tensor_pool(self, paths):
        something_rgb = self._context_rgb_pool.map(LNFGeneratorTorch._context_func_rgb, paths)
        return something_rgb[0]

    def _contextnet_labels_tensor_pool(self, paths):
        something_labels = self._context_label_pool.map(LNFGeneratorTorch._context_func_labels, paths)
        return something_labels[0]


    def transform_exp(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()])
        return composed_transforms(sample)


    # depth transformation
    def transform_tr_depth(self,sample):

            composed_transforms = transforms.Compose([
                    tr.RandomHorizontalFlip(),
                    tr.RandomCrop(crop_size=(512,512)),
                    tr.NormalizeD(mean=(0.433, 0.469, 0.408, 0.139), std=(0.187,
                                                                         0.185,
                                                                         0.178,
                                                                        0.087)),
                    tr.ToTensor()
                    ])
            return composed_transforms(sample)

    # depth transformation
    def transform_val_depth(self,sample):

            composed_transforms = transforms.Compose([
                    tr.RandomHorizontalFlip(),
                    tr.RandomCrop(crop_size=(512,512)),
                    tr.NormalizeD(mean=(0.433, 0.469, 0.408, 0.139), std=(0.187,
                                                                         0.185,
                                                                         0.178,
                                                                        0.087)),
                    tr.ToTensor()
                    ])
            return composed_transforms(sample)
    # depth transformation
    def transform_ts_depth(self,sample):

            composed_transforms = transforms.Compose([
                    tr.NormalizeD(mean=(0.433, 0.469, 0.408, 0.139), std=(0.187,
                                                                         0.185,
                                                                         0.178,
                                                                        0.087)),
                    tr.ToTensor()
                    ])
            return composed_transforms(sample)
    def transform_tr(self,sample):
            composed_transforms = transforms.Compose([
                    tr.RandomHorizontalFlip(),
                    tr.RandomCrop(crop_size=(512,512)),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor()
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
    # data preprocessing functions, these functions prepare data for training

    @staticmethod
    def _mergenet_func_rgb(path):
        im = np.asarray(Image.open(path, 'r'))
        im_cropped = im[281:793, 128:1920, :]
        return im_cropped

    @staticmethod
    def _mergenet_func_disparity(path):
        im = np.asarray(Image.open(path, 'r'))
        im = im.reshape(im.shape[0], im.shape[1], 1)
        im_cropped = im[281:793, 128:1920, :]
        im_cropped = im_cropped/256
        im_cropped = np.rint(im_cropped)
        im_cropped = im_cropped.astype(np.uint8)
        return im_cropped

    @staticmethod
    def _mergenet_func_labels(path):
        im = np.asarray(Image.open(path, 'r'))
        im_cropped = im[281:793, 128:1920].copy()
        im_cropped[im_cropped == 255] = 0
        return im_cropped


    @staticmethod
    def _context_func_rgb(path):
        im = np.asarray(Image.open(path, 'r'))
        im_cropped = im[281:793, 128:1920, :]
        return im_cropped

    @staticmethod
    def _context_func_labels(path):
        im = np.asarray(Image.open(path, 'r'))
        im_cropped = im[281:793, 128:1920].copy()
        im_cropped[im_cropped == 255] = 0
        return im_cropped


    @staticmethod
    def _func_rgb(path, stripe_size=32):
        if path[-1] == 'n':
            path = path[:-1]
            image_path = path.split('.m')[0]
            strip = int(path.split('.m')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im_cropped = im[281:793, 128:1920, :]
            return im_cropped[:, stripe_size*strip:(strip+1)*stripe_size, :]
        elif path[-1] == 'r':
            path = path[:-1]
            image_path = path.split('.n')[0]
            pixel_to_sample = int(path.split('.n')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im_cropped = im[281:793, 128:1920, :]
            return im_cropped[:, pixel_to_sample:pixel_to_sample+stripe_size, :]

    @staticmethod
    def _func_disparity(path, stripe_size=32):
        if path[-1] == 'n':
            path = path[:-1]
            image_path = path.split('.m')[0]
            strip = int(path.split('.m')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im = im.reshape(im.shape[0], im.shape[1], 1)
            im_cropped = im[281:793, 128:1920, :]

            # extract the necessary part
            disp_img = im_cropped[:, strip*stripe_size:(strip+1)*stripe_size, :]

            # normalize to rgb standards
            # convert the 16 bit depth values to 8 bit values
            disp_img = disp_img/256
            disp_img = np.rint(disp_img)
            disp_img = disp_img.astype(np.uint8)
            return disp_img

        elif path[-1] == 'r':
            path = path[:-1]
            image_path = path.split('.n')[0]
            pixel_to_sample = int(path.split('.n')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im = im.reshape(im.shape[0], im.shape[1], 1)
            im_cropped = im[281:793, 128:1920, :]
            # extract the necessary part
            disp_img = im_cropped[:, pixel_to_sample:pixel_to_sample + stripe_size, :]

            # normalize to rgb standards
            disp_img = disp_img/256
            disp_img = np.rint(disp_img)
            disp_img = disp_img.astype(np.uint8)
            return disp_img

    @staticmethod
    def _func_labels(path, stripe_size=32):
        if path[-1] == 'n':
            path = path[:-1]
            image_path = path.split('.m')[0]
            strip = int(path.split('.m')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im_cropped = im[281:793, 128:1920].copy()
            im_cropped[im_cropped == 255] = 0
            return im_cropped[:, strip*stripe_size:(strip+1)*stripe_size]
        elif path[-1] == 'r':
            path = path[:-1]
            image_path = path.split('.n')[0]
            pixel_to_sample = int(path.split('.n')[1])
            im = np.asarray(Image.open(image_path, 'r'))
            im_cropped = im[281:793, 128:1920].copy()
            im_cropped[im_cropped == 255] = 0
            return im_cropped[:, pixel_to_sample:pixel_to_sample+stripe_size]

if __name__ == "__main__":

    # this script checks for correct contruction of the image, disparity and
    # label tensors. The main check is to pass an index to the script and check
    # whether the image, disparity, label correspond to each other
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', action='store', dest='batch', type=int,
                        default=0, required=False)
    parser.add_argument('-i', action='store', dest='index', type=int,
                        default=0, required=False)
    args = parser.parse_args()
    import mypath
    # absolute_dataset_path = '/scratch/adityaRRC/small_obstacle_dataset/'
    absolute_dataset_path = mypath.Path.db_root_dir('small_obstacle')
    print(colored("Retrieving image file names...", "yellow"))
    # gets basic stripes
    train_rgbImages, train_dispImages, train_masks = get_ImagesAndLabels_from_dir(absolute_dataset_path)
    # gets additionally sampled paths around small obstacles
    train_rgbImages, train_dispImages, train_masks = generate_additional_stripes(train_rgbImages, train_dispImages, train_masks, absolute_dataset_path)
    train_generator = LNFGeneratorTorch(train_rgbImages, disparity_path=train_dispImages,
                                            mask_path=train_masks) # feed the paths in

    # x, y
    x, y = train_generator[args.batch]
    im = x[0]
    disp = x[1]
    label = y

    # store the image strip
    img = im[args.index]
    print(colored("shape of the rgb image: {}".format(img.shape),
                  "yellow"))
    result = Image.fromarray(img.astype(np.uint8))
    result.save('./visualise/debug/image_check.png')

    # store the disparity image
    disp_img = disp[args.index]
    print(colored("shape of the depth image strip: {}".format(disp_img.shape),
                 "yellow"))
    disp_img = np.reshape(disp_img, (512, 32))
    print(disp_img)
    imsave('./visualise/debug/disp_check.png', disp_img, cmap="plasma")

    # store the label image
    label_mask = label[args.index]
    print(colored("shape of the mask image: {}".format(label_mask.shape), "yellow"))
    result = Image.fromarray(label_mask.astype(np.uint8))
    lut = np.random.rand(4, 3)
    cmap = LinearSegmentedColormap.from_list('new_map', lut, N=4)

    imsave('./visualise/debug/label_check.png', result, cmap=cmap)
