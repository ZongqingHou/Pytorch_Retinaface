import torch
import torch.nn as nn
import torch.nn.functional as F

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 2)

        # out = out.view(out.shape[0], 2, -1)
        return out_


class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 4)

        # out = out.view(out.shape[0], 2, -1)
        return out_


class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)
        # self.conv1x1 = nn.Conv2d(inchannels,num_anchors*68*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 10)

        # out = out.view(out.shape[0], 2, -1)
        return out_


class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            backbone.avgpool = nn.Sequential()
            backbone.fc = nn.Sequential()

        self.body = backbone

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    @staticmethod
    def _make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        # # ---------- Training & Demo ------------------
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

        # -------------------Convert ------------------------
        # feature1 = self.ssh1(fpn[0])
        # feature2 = self.ssh2(fpn[1])
        # feature3 = self.ssh3(fpn[2])
        #
        # bbox1, class1, ldm1 = self.BboxHead[0](feature1), self.ClassHead[0](feature1), self.LandmarkHead[0](feature1)
        # bbox2, class2, ldm2 = self.BboxHead[1](feature2), self.ClassHead[1](feature2), self.LandmarkHead[1](feature2)
        # bbox3, class3, ldm3 = self.BboxHead[2](feature3), self.ClassHead[2](feature3), self.LandmarkHead[2](feature3)
        #
        # # return bbox1.view(bbox1.size(1), bbox1.size(2)), class1.view(class1.size(1), class1.size(2)), ldm1.view(ldm1.size(1), ldm1.size(2)), \
        # #        bbox2.view(bbox2.size(1), bbox2.size(2)), class2.view(class2.size(1), class2.size(2)), ldm2.view(ldm2.size(1), ldm2.size(2)), \
        # #        bbox3.view(bbox3.size(1), bbox3.size(2)), class3.view(class3.size(1), class3.size(2)), ldm3.view(ldm3.size(1), ldm3.size(2))
        #
        # return bbox1, F.softmax(class1, dim=-1), ldm1, bbox2, F.softmax(class2, dim=-1), ldm2, bbox3, F.softmax(class3, dim=-1), ldm3