import numpy as np
import cv2

'''
用于计算评价指标
'''

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # 混淆矩阵应该是 平方 这里*2代表维数
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """
        计算两个矩阵的相似度
        :param gt_image: GT
        :param pre_image: 预测
        :return:confusion_matrix（类别数*类别数）其中 index = 0 4 8 为 T ，其余false
        """
        # GT有标签（非void） 且 预测图的结果合理（没有超出3类）
        # batch_size * 512 *512
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        # 为什么要 × num_class？
        # gt_image[mask] 使得图展开
        # 利用乘法嵌入： 预测结果 0 1 2   GT 0 3 6
        # (524288,)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # 计算每个值出现的次数，常用bincount 其中 index = 0 4 8 为 T ，其余false
        count = np.bincount(label, minlength=self.num_class**2)
        # 其中 index = 0 4 8 为 T ，其余false
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def pdr_metric(self,class_id):
        '''
        计算 class_id 对应类别的准确和回归率
        :param class_id: 2-small obstacle,0-off road,1-on road
        :return: 准确和回归率
        '''
        """
        Precision and recall metric for each class
         class_id=2 for small obstacle [0-off road,1-on road]
        """
        truth_mask=self.gt_labels==class_id
        pred_mask=self.pred_labels==class_id

        true_positive=(truth_mask & pred_mask)
        true_positive=np.count_nonzero(true_positive==True)

        total=np.count_nonzero(truth_mask==True)
        pred=np.count_nonzero(pred_mask==True)

        if total != 0:
            recall=float(true_positive/total)
        else:
            recall = None

        if pred != 0:
            precision=float(true_positive/pred)
        else:
            precision = None
        return recall,precision

    def add_batch(self, gt_image, pre_image):
        """
        用于比较 GT 与 预测结果
        :param gt_image:
        :param pre_image:
        :return:混淆矩阵
        """
        assert gt_image.shape == pre_image.shape
        # 首次使用判定
        if len(self.gt_labels) == 0 and len(self.pred_labels) == 0:
            self.gt_labels=gt_image
            self.pred_labels=pre_image
        else:
            self.gt_labels=np.append(self.gt_labels,gt_image,axis=0)
            self.pred_labels=np.append(self.pred_labels,pre_image,axis=0)

        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.gt_labels=[] 
        self.pred_labels=[]

    # def add_batch(self, gt_image, pre_image):
    #     assert gt_image.shape == pre_image.shape
    #     self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    # def reset(self):
    #     self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def idr_metric(self, class_id, thresh=0.5):
        """
        计算IDR指标
        :param class_id:
        :param thresh:
        :return:
        """
        # get masks for the interested class
        truth_mask = self.gt_labels == class_id
        pred_mask = self.pred_labels == class_id

        print(truth_mask.shape)
        truth_mask = np.asarray(truth_mask, dtype=np.uint8)
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        # truth_mask = np.reshape(truth_mask, (truth_mask.shape[0],
        #                                      truth_mask.shape[1 ],
        #                                      truth_mask.shape[2], 1))
        # pred_mask = np.reshape(pred_mask, (truth_mask.shape[0],
        #                                      truth_mask.shape[1],
        #                                      truth_mask.shape[2], 1))

        labels_gt = []
        labels_pred = []
        # retrieve instances
        for i in range(len(truth_mask)):
            _, _gt = cv2.connectedComponents(np.asarray(truth_mask[i],
                                                           dtype=np.uint8), connectivity=4)
            _, _pred = cv2.connectedComponents(np.asarray(pred_mask[i],
                                                             dtype=np.uint8), connectivity=4)

            labels_gt.append(_gt)
            labels_pred.append(_pred)

        labels_gt = np.asarray(labels_gt)
        labels_pred = np.asarray(labels_pred)


        # get a list of all the unique instances to iterate through
        instance_ids = np.unique(labels_gt)
        true_positives = 0
        for instance in instance_ids:

            # ignore background instance
            if instance != 0:

                # retrieve masks for individual instances
                gt_mask = labels_gt == instance
                pred_mask = labels_pred == instance
                positives = gt_mask & pred_mask
                iou = np.sum(positives)/np.sum(gt_mask)

                # we consider an instance to be detected if the iou is greater
                # than a certain threshold
                if iou>thresh:
                    true_positives+=1

        total = len(instance_ids) - 1
        if total != 0:
            idr = true_positives/(len(instance_ids) - 1)
        else:
            idr = None
        return idr




