from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='/home/file_collections/gitlab/Face_Detection/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

args = parser.parse_args()


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
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    ## Git Repo
    # cfg = None
    # if args.network == "mobile0.25":
    #     cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     cfg = cfg_re50
    # # net and model
    # net = RetinaFace(cfg=cfg, phase = 'test')
    # net = load_model(net, args.trained_model, args.cpu)
    # net.eval()
    # print('Finished loading model!')
    # print(net)
    # device = torch.device("cpu" if args.cpu else "cuda")
    # net = net.to(device)

    # 20 pts
    # from mbn_rf_modified import Mbn_RF
    # net_dict = torch.load("/home/hdd/ppp/zhuchunbo/20_retinaface/mobilenet0.25.pth")

    # regular
    from mbn_rf import Mbn_RF
    net_dict = torch.load("/home/file_collections/gitlab/Face_Detection/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth")

    # 68 pts
    # from mbn_68pts import Mbn_RF
    # net_dict = torch.load("./weights/mobilenet0.25_epoch_795.pth")

    # 68 pts with regular mn
    # from mbn_68pts_mv1 import Mbn_RF
    # net_dict = torch.load("/home/file_collections/gitlab/Face_Detection/Pytorch_Retinaface/68pts_ply/mobilenet0.25_Final.pth")

    net = Mbn_RF()
    model_dict = {}

    for k, v in net_dict.items():
        k = k.split("module.")[-1]
        if "body." in k:
            model_dict[k.split("body.")[-1]] = v
        elif "fpn." in k:
            model_dict[k.split("fpn.")[-1]] = v
        elif "BboxHead.0." in k:
            model_dict[k.replace("BboxHead.0", "bboxhead_1")] = v
        elif "BboxHead.1." in k:
            model_dict[k.replace("BboxHead.1", "bboxhead_2")] = v
        elif "BboxHead.2." in k:
            model_dict[k.replace("BboxHead.2", "bboxhead_3")] = v
        elif "LandmarkHead.0." in k:
            model_dict[k.replace("LandmarkHead.0", "ldmhead_1")] = v
        elif "LandmarkHead.1." in k:
            model_dict[k.replace("LandmarkHead.1", "ldmhead_2")] = v
        elif "LandmarkHead.2." in k:
            model_dict[k.replace("LandmarkHead.2", "ldmhead_3")] = v
        elif "ClassHead.0." in k:
            model_dict[k.replace("ClassHead.0", "classhead_1")] = v
        elif "ClassHead.1." in k:
            model_dict[k.replace("ClassHead.1", "classhead_2")] = v
        elif "ClassHead.2." in k:
            model_dict[k.replace("ClassHead.2", "classhead_3")] = v
        elif "deconv" in k:
            pass
        else:
            model_dict[k] = v
            print(k)
            print("-------------------")

    net.load_state_dict(model_dict)
    net.eval()
    device = torch.device("cpu" if args.cpu else "cuda")

    # ------------------------ export -----------------------------
    output_onnx = 'rf_baseline_5pts_intel.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)

    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=10)


