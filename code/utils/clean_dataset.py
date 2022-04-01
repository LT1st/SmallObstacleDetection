# -*-coding:utf-8-*-

# ===============================================================================
# 用于两类数据集清洗
# sod数据集
# ===============================================================================

import os
import time
import difflib
import hashlib
import stat


def getFileMd5(filename):
    if not os.path.isfile(filename):
        print('file not exist: ' + filename)
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def getAllFiles(path):
    flist = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            f_fullpath = os.path.join(root, f)
            f_relativepath = f_fullpath[len(path):]
            flist.append(f_relativepath)
    return flist


def dirCompare(apath, bpath):
    afiles = getAllFiles(apath)
    bfiles = getAllFiles(bpath)

    setA = set(afiles)
    setB = set(bfiles)

    #     commonfiles = setA & setB  # 处理共有文件

    #     for f in sorted(commonfiles):
    #         amd=getFileMd5(apath+'\\'+f)
    #         bmd=getFileMd5(bpath+'\\'+f)
    #         if amd != bmd:
    #             print ("dif file: %s" %(f))

    # 处理仅出现在一个目录中的文件
    onlyFiles = setA ^ setB
    onlyInA = []
    onlyInB = []
    thislist = []

    for of in onlyFiles:
        if of in afiles:
            onlyInA.append(of)
        elif of in bfiles:
            onlyInB.append(of)

    if len(onlyInA) > 0:
        print('-' * 20, "only in A", apath, '-' * 20,len(onlyInA))
        for of in sorted(onlyInA):
            #print(of)
            a = apath + of
            thislist.append(a)

    if len(onlyInB) > 0:
        print('-' * 20, "only in B", bpath, '-' * 20,len(onlyInB))
        for of in sorted(onlyInB):
            #print(of)
            b = bpath + of
            thislist.append(b)

    return thislist


def deleteFile(deletelist):
    for i in deletelist:
        if os.path.exists(i):
            os.remove(i)

if __name__ == '__main__':
    from mypath import Path
    root = Path.db_root_dir('small_obstacle')
    # root = 'F:/Small_Obstacle_Dataset/'
    splits = ['train','val','test']
    deleteList = []

    for split in splits:
        trainSeq = os.listdir(os.path.join(root, split))
        for seq in trainSeq:
            imagePath = os.path.join(root, split, seq, 'image')
            labelPath = os.path.join(root, split, seq, 'labels')

            # labelPath = r'F:\Small_Obstacle_Dataset\train\file_3\labels'
            # imagePath = r'F:\Small_Obstacle_Dataset\train\file_3\image'
            #
            tmp = dirCompare(imagePath, labelPath)
            deleteList.extend(tmp)

    print(deleteList)
    # import pickle
    # pickle.dump(deleteList,open('deletefiles','wb'))
    file = open('C:/Users/LUTAO11/Desktop/deletefiles.txt','w')
    for i in deleteList:
        #print(type(i))
        file.write(i)
    file.close()

    # thisf = 'F:/Small_Obstacle_Dataset/train/file_2/image/0000000035.png'
    # os.chmod(thisf, stat.S_IRWXO)
    # deleteFile(thisf)


    for i in deleteList:
        # os.path.normpath(i)
        # i.replace("\\", "/")
        path, name = os.path.split(os.path.normpath(i))

        os.chdir(path)
        # 提升权限
        os.chmod(name, stat.S_IRWXU)
        # os.chmod(name, stat.S_IRWXO)

        print(i)
        # PermissionError: [WinError 5] 拒绝访问。: '/'
        os.remove(name)
    # aPath = r'F:\Small_Obstacle_Dataset\train\seq_6\labels'
    # bPath = r'F:\Small_Obstacle_Dataset\train\seq_6\image'
    # dirCompare(aPath, bPath)
    print("\ndone!")
