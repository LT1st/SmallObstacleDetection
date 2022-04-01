import platform

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        '''
        集中管理数据集地址
        :param dataset: 数据集名称
        :return: 数据集根地址
        '''
        sysType = platform.system()
        print("u are running in : {}".format(sysType))
        # windows 下的路径（注意\为转译符）
        if sysType == 'Windows':
            if dataset == 'pascal':
                return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            elif dataset == 'sbd':
                return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
            elif dataset == 'cityscapes':
                return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
            elif dataset == 'coco':
                return '/path/to/datasets/coco/'
            elif dataset == 'small_obstacle':
                return 'F:/Small_Obstacle_Dataset/'
            elif dataset == 'lnf':
                return 'F:/lostAndFound/'
            else:
                print('Dataset {} not available.'.format(dataset))
                raise NotImplementedError
        # linux下的路径
        else :
            if dataset == 'pascal':
                return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            elif dataset == 'sbd':
                return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
            elif dataset == 'cityscapes':
                return '/dataset/public/cityscapes/'     # foler that contains leftImg8bit/
            elif dataset == 'coco':
                return '/path/to/datasets/coco/'
            elif dataset == 'small_obstacle':
                return '/data1/dataset/Small_Obstacle_Dataset/'
            elif dataset == 'lnf':
                # ？？？这downtown_data_run到底是啥
                return '/data1/dataset/LostAndFound/'
            else:
                print('Dataset {} not available.'.format(dataset))
                raise NotImplementedError