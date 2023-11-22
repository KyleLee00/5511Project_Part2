import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import  sys

from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *
from models.mobilenetv2 import MobileNetV2

from loss.loss import *
import cv2
import time
import json
from datetime import datetime
import random

def trainer(ops,f_log):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        if ops.log_flag:
            sys.stdout = f_log

        set_seed(ops.seed)
        model_ = MobileNetV2(num_classes=ops.num_classes , dropout_factor = ops.dropout)


        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        print(model_)
        # Dataset
        dataset = LoadImagesAndLabels(ops= ops,img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,vis = False)
        print("loading done")

        print('length of train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # optimizer
        optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99),weight_decay=1e-6)
        optimizer = optimizer_Adam
        #  finetune model
        if os.access(ops.fintune_model,os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location=device)
            model_.load_state_dict(chkpt)
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')
        # loss function
        if ops.loss_define == 'mse_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # parameter
        best_loss = np.inf
        loss_mean = 0.
        loss_idx = 0.
        flag_change_lr_cnt = 0
        init_lr = ops.init_lr

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 50:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0.
            loss_idx = 0.

            for i, (imgs_, pts_) in enumerate(dataloader):
                if use_cuda:
                    imgs_ = imgs_.cuda()
                    pts_ = pts_.cuda()

                output = model_(imgs_.float())
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_.float())
                else:
                    loss = criterion(output, pts_.float())
                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                    ' lr : %.8f'%init_lr,' bs :',ops.batch_size,\
                    ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss: %.6f'%best_loss, " {}".format(ops.loss_define))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            #set_seed(random.randint(0,65535))
            torch.save(model_.state_dict(), ops.model_exp + '{}-size-{}-loss-{}-model_epoch-{}.pth'.format(ops.model,ops.img_size[0],ops.loss_define,epoch))

    except Exception as e:
        print('Exception : ',e)
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])
        print('Exception  line : ', e.__traceback__.tb_lineno)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Handpose Train')
    parser.add_argument('--seed', type=int, default = 114514,
        help = 'seed')
    parser.add_argument('--model_exp', type=str, default = './model_exp',
        help = 'model_exp')
    parser.add_argument('--model', type=str, default = 'mobilenetv2')
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes')
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS')

    parser.add_argument('--train_path', type=str,
        default = "./handpose_datasets_v1/",
        help = 'datasets')

    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain')
    parser.add_argument('--fintune_model', type=str, default = './weights/mobilenetv2-size-256-loss-wing_loss-model_epoch-9.pth',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = 'wing_loss',
        help = 'define_loss : wing_loss, mse_loss ')
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init learning Rate')
    parser.add_argument('--lr_decay', type=float, default = 0.1,
        help = 'learningRate_decay')
    parser.add_argument('--weight_decay', type=float, default = 1e-6,
        help = 'weight_decay')
    parser.add_argument('--momentum', type=float, default = 0.9,
        help = 'momentum')
    parser.add_argument('--batch_size', type=int, default = 16,
        help = 'batch_size')
    parser.add_argument('--dropout', type=float, default = 0.5,
        help = 'dropout') # dropout
    parser.add_argument('--epochs', type=int, default = 10,
        help = 'epochs')
    parser.add_argument('--num_workers', type=int, default = 10,
        help = 'num_workers')
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size')
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation')
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution')
    parser.add_argument('--clear_model_exp', type=bool, default = False,
        help = 'clear_model_exp')
    parser.add_argument('--log_flag', type=bool, default = True,
        help = 'log flag')

    #--------------------------------------------------------------------------
    args = parser.parse_args()
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = True
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args)
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args,f_log = f_log)

    if args.log_flag:
        sys.stdout = f_log
    print('Training done. : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
