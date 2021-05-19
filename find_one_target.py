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
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='/home/intern/zhuchunbo/retinaface/weights/Resnet50_Final_68.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
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
    model.load_state_dict(pretrained_dict, strict=True)
    return model


if __name__ == '__main__':

    # path = '/home/intern/zhuchunbo/retinaface/dataset/rs_voc/'
    path = '/home/intern/zhuchunbo/retinaface/widerface/train/images/realsence/'
    rightpath='/home/intern/zhuchunbo/retinaface/dataset/realsence/'
    errorpath='/home/intern/zhuchunbo/retinaface/dataset/error/'
    id = path.split('/')[3]
    f_error = open('/home/intern/zhuchunbo/retinaface/dataset/error.txt', 'w')

    torch.set_grad_enabled(False)
    cfg = cfg_re50
    # if args.network == "mobile0.25":
    #     cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    resize = 1
    cap = cv2.VideoCapture(0)

    files = os.listdir(path)
    files = np.sort(files)
    i = 1
    f5 = open('bbox.txt', 'w')
    for f in files:
        imgpath = path + f
        if imgpath.endswith("rgb.png"):
            img_raw0 = cv2.imread(imgpath, cv2.IMREAD_COLOR)

            img_raw = cv2.resize(img_raw0, (640, 640))
            img = np.float32(img_raw)
            im_height, im_width, _ = img.shape

            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = net(img)

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2],])
            scale1=scale1.repeat(16800,68)
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # if dets.size!=68*2+5:#only one face has been detected
            #     cv2.imwrite(errorpath + f, img_raw0)
            #     f_error.write(f+'\n')
            # else:
            #     cv2.imwrite(rightpath + f, img_raw0)


            # if dets.size == 68 * 2 + 5:
            for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    im_height0, im_width0, _ = img_raw0.shape
                    bbox_x = str(b[0] * im_width0 / im_width)
                    bbox_y = str(b[1] * im_height0 / im_height)
                    bbox_w = str(b[2] * im_width0 / im_width - b[0] * im_width0 / im_width)
                    bbox_h = str(b[3] * im_height0 / im_height - b[1] * im_height0 / im_height)

                    # x = int(b[0] * im_width0 / 640)
                    # y = int(b[1] * im_height0 / 640)
                    # w = int(b[2] * im_width0 / 640)
                    # h = int(b[3] * im_height0 / 640)
                    # cv2.rectangle(img_raw0, (x, y), (w, h), (0, 0, 255), 2)
                    # cv2.imwrite(rightpath + 'result' + f, img_raw0)

                    file_name = (os.path.basename(path+f)).split('.')[0]
                    f5.write('# ' + 'realsence/' +file_name + '.png' + '\n')
                    f5.write(bbox_x + ' ' + bbox_y + ' ' + bbox_w + ' ' + bbox_h + ' ')
                    f5.write('\n')

            print(i)
            i = i + 1
