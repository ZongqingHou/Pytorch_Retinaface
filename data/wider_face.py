import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
<<<<<<< HEAD

=======
from torchvision import transforms as tfs
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
<<<<<<< HEAD
                self.imgs_path.append(path)
=======
                # print(path)
                self.imgs_path.append(path)
                # print(self.imgs_path)
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
<<<<<<< HEAD
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
=======
        annotations = np.zeros((0, 141))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 141))
>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

<<<<<<< HEAD
            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
=======
            # 30landmarks
            # annotation[0, 4] = label[10]    # l0_x
            # annotation[0, 5] = label[11]    # l0_y
            # annotation[0, 6] = label[19]    # l1_x
            # annotation[0, 7] = label[20]    # l1_y
            # annotation[0, 8] = label[25]   # l2_x
            # annotation[0, 9] = label[26]   # l2_y
            # annotation[0, 10] = label[37]  # l3_x
            # annotation[0, 11] = label[38]  # l3_y
            # annotation[0, 12] = label[49]  # l4_x
            # annotation[0, 13] = label[50]  # l4_y
            # annotation[0, 14] = label[52]
            # annotation[0, 15] = label[53]
            # annotation[0, 16] = label[58]
            # annotation[0, 17] = label[59]
            # annotation[0, 18] = label[64]
            # annotation[0, 19] = label[65]
            # annotation[0, 20] = label[67]
            # annotation[0, 21] = label[68]
            # annotation[0, 22] = label[73]
            # annotation[0, 23] = label[74]
            # annotation[0, 24] = label[79]
            # annotation[0, 25] = label[80]
            # annotation[0, 26] = label[109]
            # annotation[0, 27] = label[110]
            # annotation[0, 28] = (float(label[112])+float(label[115]))/2
            # annotation[0, 29] = (float(label[113])+float(label[116]))/2
            # annotation[0, 30] = label[118]
            # annotation[0, 31] = label[119]
            # annotation[0, 32] = (float(label[121])+float(label[124]))/2
            # annotation[0, 33] = (float(label[122])+float(label[125]))/2
            # annotation[0, 34] = label[127]
            # annotation[0, 35] = label[128]
            # annotation[0, 36] = (float(label[130])+float(label[133]))/2
            # annotation[0, 37] = (float(label[131])+float(label[134]))/2
            # annotation[0, 38] = label[136]
            # annotation[0, 39] = label[137]
            # annotation[0, 40] = (float(label[139])+float(label[142]))/2
            # annotation[0, 41] = (float(label[140])+float(label[143]))/2
            # annotation[0, 42] = label[82]
            # annotation[0, 43] = label[83]
            # annotation[0, 44] = label[88]
            # annotation[0, 45] = label[89]
            # annotation[0, 46] = label[94]
            # annotation[0, 47] = label[95]
            # annotation[0, 48] = label[100]
            # annotation[0, 49] = label[101]
            # annotation[0, 50] = label[106]
            # annotation[0, 51] = label[107]
            # annotation[0, 52] = label[145]
            # annotation[0, 53] = label[146]
            # annotation[0, 54] = label[154]
            # annotation[0, 55] = label[155]
            # annotation[0, 56] = label[163]
            # annotation[0, 57] = label[164]
            # annotation[0, 58] = label[172]
            # annotation[0, 59] = label[173]
            # annotation[0, 60] = label[187]
            # annotation[0, 61] = label[188]
            # annotation[0, 62] = label[199]
            # annotation[0, 63] = label[200]

            # 20landmarks
            # annotation[0, 4] = label[10]#2
            # annotation[0, 5] = label[11]
            # annotation[0, 6] = label[19]#5
            # annotation[0, 7] = label[20]
            # annotation[0, 8] = label[28]#8
            # annotation[0, 9] = label[29]
            # annotation[0, 10] = label[37]#11
            # annotation[0, 11] = label[38]
            # annotation[0, 12] = label[46]#14
            # annotation[0, 13] = label[47]
            # annotation[0, 14] = label[55]#17
            # annotation[0, 15] = label[56]
            # annotation[0, 16] = label[61]#19
            # annotation[0, 17] = label[62]
            # annotation[0, 18] = label[67]#21
            # annotation[0, 19] = label[68]
            # annotation[0, 20] = label[70]#22
            # annotation[0, 21] = label[71]
            # annotation[0, 22] = label[76]#24
            # annotation[0, 23] = label[77]
            # annotation[0, 24] = label[82]#26
            # annotation[0, 25] = label[83]
            # annotation[0, 26] = label[112]#36
            # annotation[0, 27] = label[113]
            # annotation[0, 28] = label[121]#39
            # annotation[0, 29] = label[122]
            # annotation[0, 30] = label[130]#42
            # annotation[0, 31] = label[131]
            # annotation[0, 32] = label[139]#45
            # annotation[0, 33] = label[140]
            # annotation[0, 34] = label[85]#27
            # annotation[0, 35] = label[86]
            # annotation[0, 36] = label[91]#29
            # annotation[0, 37] = label[92]
            # annotation[0, 38] = label[103]#33
            # annotation[0, 39] = label[104]
            # annotation[0, 40] = label[148]#48
            # annotation[0, 41] = label[149]
            # annotation[0, 42] = label[166]#54
            # annotation[0, 43] = label[167]


            # 68 landmarks
            for i in range(4,140,2):
                annotation[0,i]=label[int(i+i/2-2)]
                annotation[0,i+1]=label[int(i+1+i/2-2)]

            if (annotation[0, 4]<0):
                annotation[0, 140] = -1
            else:
                annotation[0, 140] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        # print(self.imgs_path[index])

        # data augment use transforms in torchvision

>>>>>>> d6ac046416eb6c4f55ad26685f55ba7908d4a901
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
