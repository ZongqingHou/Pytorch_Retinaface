import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.nn.functional as F

=======
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


<<<<<<< HEAD
=======

>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
<<<<<<< HEAD
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 2)

        # out = out.view(out.shape[0], 2, -1)
        return out_

=======
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
<<<<<<< HEAD
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 4)

        # out = out.view(out.shape[0], 2, -1)
        return out_

=======
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

# class LandmarkHead(nn.Module):
#     def __init__(self,inchannels=512,num_anchors=3):
#         super(LandmarkHead,self).__init__()
#         self.conv1x1 = nn.Conv2d(inchannels,num_anchors*68*2,kernel_size=(1,1),stride=1,padding=0)
#
#     def forward(self,x):
#         out = self.conv1x1(x)
#         out = out.permute(0,2,3,1).contiguous()
#         return out.view(out.shape[0], -1, 68*2)
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
<<<<<<< HEAD
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)
        # self.conv1x1 = nn.Conv2d(inchannels,num_anchors*68*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out_ = out.permute(0,2,3,1).contiguous()
        out_ = out_.view(out_.shape[0], -1, 10)

        # out = out.view(out.shape[0], 2, -1)
        return out_

=======
        self.conv_part1 = nn.Conv2d(inchannels,num_anchors*17*2,kernel_size=(1,1),stride=1,padding=0)
        self.conv_part2 = nn.Conv2d(inchannels,num_anchors*10*2,kernel_size=(1,1),stride=1,padding=0)
        self.conv_part3 = nn.Conv2d(inchannels,num_anchors*12*2,kernel_size=(1,1),stride=1,padding=0)
        self.conv_part4 = nn.Conv2d(inchannels,num_anchors*9*2,kernel_size=(1,1),stride=1,padding=0)
        self.conv_part5 = nn.Conv2d(inchannels,num_anchors*20*2,kernel_size=(1,1),stride=1,padding=0)
    def forward(self,x):
        out_part1 = self.conv_part1(x)
        out_part2 = self.conv_part2(x)
        out_part3 = self.conv_part3(x)
        out_part4 = self.conv_part4(x)
        out_part5 = self.conv_part5(x)

        out_part1_1, out_part1_2 = torch.chunk(out_part1, 2, dim=1)
        out_part2_1, out_part2_2 = torch.chunk(out_part2, 2, dim=1)
        out_part3_1, out_part3_2 = torch.chunk(out_part3, 2, dim=1)
        out_part4_1, out_part4_2 = torch.chunk(out_part4, 2, dim=1)
        out_part5_1, out_part5_2 = torch.chunk(out_part5, 2, dim=1)

        output_1 = torch.cat((out_part1_1, out_part2_1, out_part3_1, out_part4_1, out_part5_1), dim=1)
        output_2 = torch.cat((out_part1_2, out_part2_2, out_part3_2, out_part4_2, out_part5_2), dim=1)
        output = torch.cat((output_1, output_2), dim=1)
        output = output.permute(0,2,3,1).contiguous()
        return output.view(output.shape[0], -1, 68*2)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict =remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=True)
    return model
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901

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
<<<<<<< HEAD
            backbone = MobileNetV1()
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            backbone.avgpool = nn.Sequential()
            backbone.fc = nn.Sequential()

        self.body = backbone

=======
            backbone=MobileNetV1()
            # backbone = load_model(basenet, './weights/mobilenet0.25_Final_68.pth',cfg['pretrain'])

            # if cfg['pretrain']:
            #     checkpoint = torch.load("./weights/mobilenet0.25_Final_68.pth")
            #     from collections import OrderedDict
            #     new_state_dict = OrderedDict()
            #     for k, v in checkpoint['state_dict'].items():
            #         name = k[7:]  # remove module.s
            #         new_state_dict[name] = v
            #     # load params
            #     backbone.load_state_dict(new_state_dict)

        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
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

<<<<<<< HEAD
    @staticmethod
    def _make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
=======
    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
<<<<<<< HEAD

    @staticmethod
    def _make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
=======
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

<<<<<<< HEAD
    @staticmethod
    def _make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
=======
    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

<<<<<<< HEAD
    def forward(self, inputs):
=======
    def forward(self,inputs):
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
<<<<<<< HEAD
        # # ---------- Training & Demo ------------------
=======
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
<<<<<<< HEAD
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
=======
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
<<<<<<< HEAD
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
=======
        return output
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
