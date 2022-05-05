# SmallObstacleDetection
Small Obstacle Detection imlemetation
代码在子文件夹下

# Deeplab-V3-Pytorch: Small Obstacle 

### TODO
- [x] Modify for 4 channel input 

### Training
Follow below steps below to train your model:

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val][--mode]
    ```
   
# 代码注意事项
1. 数据集需要在mypath中替换
2. DRN的网络pretrain需要在新的webroot下载，原链接失效
```
   webroot = 'http://dl.yf.io/drn/'
```
3. saver的函数有问题，
   - [x] 改一下saver.py
4. small_obstacle的append路徑需要改一下
5. writer的输入channel错误，需要按照mode来修改  
   输入为： 训练 pair ， channel ， 图像分辨 x*y
   - [x] 根据 depth 修改 tensor = torch.zeros([batch_size, channels, 512, 512])
6. 检查所有default没有的args，防止漏参数  
   depth & mode 是重点  
7. - [x] 数据集没写好  
8. 作者的readme是空的，需要自己探索 
8. condalist是本人运行环境，可以跑  

| 论文数据集 | 代码用的 | 
| ---- | ----|
|  cityscapes：pretrain| LF/CS |
|small_obstacle：fine tune|  LF/CS  |
8. utils/helper.py用于数据加载，有点阴间。dataloader是抄的，有条纹方法残留。阅读比较成熟的LF等代码会有帮助。
9. 部分代码batchsize写死了，2，得手动换一下。看到的都改了
10. 代码没写完WDNMD，matchTemplate全项目都没用到。squeezeseg没有相关代码。
11. 文中context指的是： RGB+lidar 
12. 部分算法使用jupyter notebook写好了再移植，如遇bug，可以试试notebook找一下


## 路径问题
* misc.py 
* 尽量使用mypath.Path.db_root_dir('small_obstacle')获取路径
* 不清楚具体路径是作者私有数据集结构与开放不同，还是传递结构不同，还是写错了
* 注意在系统路径中append当前目录

# 其他
baseline源于 Deeplabv3Plus-Pytorch

# TODO
-[x] 搞清楚用的这几个数据集  
      搞不明白，耦合严重，LF\CS数据集改来的，感觉没改完（如标签加载那边）。其他的数据集没写。自己假设一个。
-[x] 做一个文件结构，整明白具体方法
-[X] SOD数据集跑起来
-[x] 训练流程明确一下

# 日志
| 报错 | 错因 | 解决 |
| ---- | ----| ----|
| 无法使用linux运行test | 验证集数据无法加载| 未知问题， 
| 给模型输入一张图片时[1, 256, 1, 1]报错 | BN操作的要求 | 法一 1036张图片，手动调整BS，防止其余1
|  |  | 法二 dataloader的drop_last=True
| 无法使用val，总是最后一个epoch报错 | 没写好，有时间debug | 
| 数据集读取报错 | F:\Small_Obstacle_Dataset\train\seq_5\image 的27之前没有label | 写了一个删除脚本，去掉没有label的img和没有img的label
|windows删除没有权限 | os.remove权限不够 |在属性-安全中调整
| | |os.chmod(i, stat.S_IRWXU)

# 訓練流程
## 融合（二选一）
### 使用点云
1. -[x] lidar檢測  照着写一下，传统算法那个
2. -[x] 檢測結果过滤  角度上的过滤大障碍物x。道路分割过滤道路外障碍物。
3. -[x] 检测结果 点云转换 到图像平面
0. -[x] 图像平面 高斯模糊 为置信度图 （论文中说是作为第四通道RGBA）。但这样下一步是做什么的呢？
0. -[x] 置信度图转化为RGB？？
5. -[x] 获取深度图
   
### 使用深度图
1. -[x] 利用标签，给深度图mask，获取ROI区域的深度点
2. -[x] 执行高斯模糊
3. -[x] 作为第四维度拼接 (发现数据集BUG，标签缺失太多，无法对齐)
   
## 训练
1. -[x] RGB数据集预训练(注意这里是RGB，脚本为
```Shell
python train.py --dataset catyscapes 
```
同时，参考作者的几个sh文件，需要用两阶段调整学习率的方法训练
2. -[x] 加入深度，在small_obs上训练
```Shell
python train.py --dataset small_obstacle --depth 
```
3. linux服务器上的完整脚本的格式如下：
```shell
cd /data1/data_test/small_obstacle_discovery-master-linux; mkdir /root/.cache/;mkdir  /root/.cache/torch/; mkdir /root/.cache/torch/hub/; mkdir /root/.cache/torch/hub/checkpoints/; cp ../drn_d_54-0e0534ff.pth /root/.cache/torch/hub/checkpoints/drn_d_54-0e0534ff.pth ; pip install tensorboardX;python train.py --epochs 80 --batch-size 16 --mode train --dataset cityscapes --logsFlag Cityscapes --no-val
```
# 推理流程
## 檢測（生成置信度圖）（由lidar转换为置信度）
### 使用点云
1. -[x] lidar檢測(这个必须在匹配之前跑，这里在原始点云文件的5维上额外加了障碍物判定的四个维度)
1. -[ ] 道路分割（lidar或rgb都可以，来不及了）
2. -[ ] 檢測結果过滤（道路结果作为mask，点乘即可）
3. -[x] 检测结果 点云转换 到图像平面
4. -[x] 图像平面 高斯模糊 为置信度图 （论文中说是作为第四通道RGBA）。但这样下一步是做什么的呢？
4. -[x] 置信度图转化为RGB？？
5. -[x] 获取深度图
### 使用深度图
1. -[x] 利用标签，给深度图mask，获取ROI区域的深度点
2. -[x] 执行高斯模糊（没给参数）
3. -[x] 作为当前帧置信度图
4. -[ ] 需要加入大概率的随机擦除，不然过拟合严重（相当于是label泄露），依赖lidar数据很重
   
## 跟蹤(置信度的时间传播)
1. -[x] 选取先前帧中的ROI区域
1. -[x] 当前帧中进行模板匹配
3. -[ ] 利用惯性导航数据加速搜索?????
4. -[ ] 得到修正后的置信度图
   
## 分割
1. -[x] 置信度图结合RGB输入，執行分割
   
# 数据集
## small obstacle dataset （SOD）（IIIT）
|标签|名称
| ---- | ---- |
|0| 道路外
|1| 道路
|>1| 障碍物，9个具体类 |
|5| 箱子
## Lost and Found （lnf）（LF）
成熟了，网上多
## cityscapes
成熟了，网上多

# 2022.2.26状态

## 发现但是没解决的BUG
| BUG | 尝试的解决方法 | 跳过方式 |
| --- | --- | --- | 
| 代码在windows上ok，但是在linux的val部分，会卡死在最后一个epoch| 放弃最后一个epoch | 使用--no-val，模型训练完了再验证
| epoch=1时。网络结构的BN操作报错 |  | dataloader的drop_last=True 或者 计算好epoch，不落单
| 高斯模糊参数未知 | 消融实验没发现有啥变化 | 未知
| small obstacle的数据集加载部分，速度可优化50%以上（参考LF数据集） |  |
| 多卡计算速度异常，两张卡速度和一张差不多 |  |
| cityscapes预训练来不及做 | | 用的是LNF做预训练
| 为了算法可读性，保留了冗余变量 ||
| 环带检测跳变的计算距离中使用什么范数好| 没做实验 |
| 环带检测跳变中，边界点未处理（两处）||
| 功能来不及整合，点云中的障碍物检测（utils/lidar.py）||
| 功能来不及整合，历史帧匹配（utils/match_Template.py）||
| 功能不完善，历史帧匹配（utils/match_Template.py）需要多模板匹配多目标，模板匹配需要多尺度、NMS等提升acc，需要改写为卷积提升效率||
