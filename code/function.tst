│  function.md
│  misc.py  /downtown_data_run/数据集加载器
│  mypath.py  路径管理
    def db_root_dir(dataset):
        '''
        集中管理数据集地址
        :param dataset: 数据集名称
        :return: 数据集根地址
        '''
│  normalization_metrics.py
    """
    计算数据集的 mean 和 std
    """
│  train.py
│
├─dataloaders
│  │  custom_transforms.py
│  │  utils.py
│  │  __init__.py
│  │
│  ├─datasets
│  │  │  cityscapes.py      本文使用的pretrain数据集
│  │  │  coco.py
│  │  │  combine_dbs.py
│  │  │  exploremydata.py   探索LF数据集
│  │  │  findMyData.py
│  │  │  pascal.py
│  │  │  sbd.py
│  │  │  small_obstacle.py
│  │  │  __init__.py
├─doc
│      deeplab_resnet.py
│      deeplab_xception.py
│      results.png
│
├─docs      数据集介绍网页
├─logs      日志
├─modeling  构建模型
│  │  aspp.py       ASPP结构
│  │  decoder.py    解码器
│  │  deeplab.py    deeplabV3+
│  │  __init__.py
│  │
│  ├─backbone        骨干网
│  ├─sync_batchnorm  多卡同步BN
│
├─utils     各种工具
│  │  calculate_weights.py  计算权重
│  │  encode_segmap.py      编码分割图
│  │  helpers.py            LF数据集加载工具
│  │  loss.py               本文使用的损失函数
│  │  lr_scheduler.py       学习率管理器
│  │  metrics.py            衡量
│  │  saver.py              保存日志
│  │  summaries.py          tensorboard的总结
