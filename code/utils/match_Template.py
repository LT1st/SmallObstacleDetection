import numpy as np
import cv2



def getROI(segResult, pastImg):
    """
    根据以前帧lidar检测结果搜索闭合区域，获取BBOX，用BBOX抠图
    :param segResult: 图像检测的结果，np.array格式，sahpe为(点数, 9)
    :param pastImg: 被搜索的RGB
    :return:RGB的ROI区域，可能有多个，这里假设闭合区域为1，以后再改
    """
    ROI = []
    # 获取分割结果为正的
    mask = segResult[:, 8] == 1
    xy = segResult[:, 0:2][mask]

    # 用xy 求一下联通区
    # todo

    xmin = xy[:,0].min()
    xmax = xy[:,0].max()
    ymin = xy[:,1].min()
    ymax = xy[:,1].max()

    # 这里需要给范围append一下，并且注意不能超出图像区域
    # todo

    # 获取ROI
    ROI.append(pastImg[xmin:xmax, ymin:ymax,:])

    return ROI



def findOBSinRGB(ROIs, img,threshold=0.8):
    """
    输入上一帧的障碍物ROI
    在本帧匹配寻找ROI
    如果匹配度大于阈值，返回位置区域
    注意位置区域可能需要稀疏化
    :return: 本帧匹配ROI的几个点（n,x,y）
    # 需要增加对于多ROI支持，ROI规范为list
    """
    loc = []
    for ROI in ROIs:

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(img_gray, ROI)
        loc.append(np.where(res >= threshold))

    return loc
