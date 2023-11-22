import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
from models.mobilenetv2 import MobileNetV2


from utils.common_utils import *
import copy
from hand_data_iter.datasets import draw_bd_handpose

# set parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_path', type=str, default = './weights/mobilenetv2-size-256-loss-wing_loss-model_epoch-9.pth',
        help = 'model_path')
    parser.add_argument('--model', type=str, default = 'mobilenetv2')
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes')
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS')
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path')
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size')
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis')

    print('\n/******************* {} ******************/\n'.format(parser.description))
    ops = parser.parse_args()
    print('----------------------------------')

    unparsed = vars(ops)
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    test_path =  ops.test_path

    print('use model : %s'%(ops.model))
    model_ = MobileNetV2(num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()

    print(model_)

    # load test model
    if os.access(ops.model_path,os.F_OK):
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    with torch.no_grad():
        idx = 0
        for file in os.listdir(ops.test_path):
            if '.jpg' not in file:
                continue
            idx += 1
            print('{}) image : {}'.format(idx,file))
            img = cv2.imread(ops.test_path + file)
            img_width = img.shape[1]
            img_height = img.shape[0]

            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()
            pre_ = model_(img_.float())
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            pts_hand = {}
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(img,pts_hand,0,0)

            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                cv2.circle(img, (int(x),int(y)), 3, (255,50,60),-1)
                cv2.circle(img, (int(x),int(y)), 1, (255,150,180),-1)

            # save image
            result_image = img.copy()
            for i in range(int(output.shape[0] / 2)):
                x = int(output[i * 2 + 0] * float(img_width))
                y = int(output[i * 2 + 1] * float(img_height))
                cv2.circle(result_image, (x, y), 3, (255, 50, 60), -1)
                cv2.circle(result_image, (x, y), 1, (255, 150, 180), -1)

            save_path = os.path.join(test_path, f"result_{idx}.jpg")
            cv2.imwrite(save_path, result_image)

            if ops.vis:
                cv2.namedWindow('image',0)
                cv2.imshow('image',img)
                if cv2.waitKey(600) == 27 :
                    break

    cv2.destroyAllWindows()

    print('inference done.')
