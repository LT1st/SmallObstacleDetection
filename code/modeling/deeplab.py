import torch
import sys
sys.path.append('/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/')
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
        def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                                 sync_bn=True, freeze_bn=False, depth=False):
                super(DeepLab, self).__init__()
                if backbone == 'drn':
                        output_stride = 8

                if sync_bn == True:
                        BatchNorm = SynchronizedBatchNorm2d
                else:
                        BatchNorm = nn.BatchNorm2d

                print("DeepLab constructor:", depth)
                self.backbone = build_backbone(backbone, output_stride,
                                               BatchNorm, depth)
                self.aspp = build_aspp(backbone, output_stride, BatchNorm)
                self.decoder = build_decoder(num_classes, backbone, BatchNorm)

                if freeze_bn:
                        self.freeze_bn()

        def forward(self, input):
                # 输入batch
                # low_level_feat ： torch.Size([2, 256, 128, 128])
                x, low_level_feat = self.backbone(input)
                # ASPP 2*3*256
                x = self.aspp(x)
                # change related to uncertainty
                # 置信度图和输出来源一样，就是1*1卷积核数不一样
                # 2*3*128*128  4X的下率
                x, conf = self.decoder(x, low_level_feat)
                # 4X的上采样
                pre_conf = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

                # change related to uncertainty
                conf = F.interpolate(conf, size=input.size()[2:], mode='bilinear', align_corners=True)

                # change related to uncertainty
                conf = F.sigmoid(conf)
                # 这是什么操作？
                x = pre_conf*conf

                return x, conf, pre_conf

        def freeze_bn(self):
                for m in self.modules():
                        if isinstance(m, SynchronizedBatchNorm2d):
                                m.eval()
                        elif isinstance(m, nn.BatchNorm2d):
                                m.eval()

        def get_1x_lr_params(self):
                modules = [self.backbone]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p

        def get_10x_lr_params(self):
                modules = [self.aspp, self.decoder]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p


if __name__ == "__main__":
        model = DeepLab(backbone='drn', output_stride=16,num_classes=3,
                        depth=True)
        model.eval()
        input = torch.rand(1, 4, 512, 512)
        output = model(input)
        checkpoint=torch.load('/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-drn.pth.tar',map_location='cpu')
        weight_shape=[3,256,1,1]
        conf_weight_shape=[1,256,1,1]
        checkpoint['state_dict']['decoder.diverge_conv_pred.weight']=nn.init.kaiming_normal_(torch.empty(weight_shape))
        checkpoint['state_dict']['decoder.diverge_conv_pred.bias']=nn.init.constant_(torch.empty(weight_shape[0]),0)
        checkpoint['state_dict']['decoder.diverge_conv_conf.weight']=nn.init.kaiming_normal_(torch.empty(conf_weight_shape))
        checkpoint['state_dict']['decoder.diverge_conv_conf.bias']=nn.init.constant_(torch.empty(conf_weight_shape[0]),0)
        del checkpoint['state_dict']['decoder.last_conv.8.weight']
        del checkpoint['state_dict']['decoder.last_conv.8.bias']
        f=open("/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-image_input_uncertainty.pth","wb")
        torch.save(checkpoint,f)
        f.close()
        # for i,param in enumerate(checkpoint['state_dict']):
        #         print(param," shape", checkpoint['state_dict'][param].shape)
        depth_weight_shape = [16,1,7,7]
        depth_channel_weights = nn.init.kaiming_normal_(torch.empty(depth_weight_shape))
        temp = torch.cat((checkpoint['state_dict']['backbone.layer0.0.weight'],
                  depth_channel_weights), 1)
        print(temp.shape)
        checkpoint['state_dict']['backbone.layer0.0.weight'] = temp
        f=open("/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-depth_input_uncertainty.pth","wb")
        torch.save(checkpoint,f)
        f.close()
