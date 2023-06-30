import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import time
import sys
import h5py
from networks.net_factory_3d import net_factory_3d
import math
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--imgs_infer_list', type=str,default='dateset/unet_3D/fold0_big/fold0train01_remain09.csv',
                    help='the data to infer list(name list)')
parser.add_argument('--weight_path', type=str,default='checkpoint/unet_3D_past/09/unet_3D_best_model.pth',)
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']= args.devicenum

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Building model..')
net = net_factory_3d(net_type='unet_3D', in_chns=1, class_num=2)
# net = torch.nn.DataParallel(net)
state_dict = torch.load(args.weight_path)
net.load_state_dict(state_dict)
net = net.to(device)
net.eval()

criterion = nn.MSELoss().to(device)

def test_single_case_for_consis_loss(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    loss_this_data = 0
    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_1 = np.flip(test_patch, 0)
                # test_patch_2 = np.flip(test_patch, 1)
                test_patch_2 = np.flip(test_patch, 1)
                test_patch_3 = np.flip(np.flip(test_patch, 0), 1)
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)#在batch/c两个部分增加维度，bc h w d
                test_patch_1 = np.expand_dims(np.expand_dims(test_patch_1, axis=0), axis=0).astype(np.float32)
                test_patch_2 = np.expand_dims(np.expand_dims(test_patch_2, axis=0), axis=0).astype(np.float32)
                test_patch_3 = np.expand_dims(np.expand_dims(test_patch_3, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch_1 = torch.from_numpy(test_patch_1).cuda()
                test_patch_2 = torch.from_numpy(test_patch_2).cuda()
                test_patch_3 = torch.from_numpy(test_patch_3).cuda()

                with torch.no_grad():
                    y = net(test_patch)
                    y1 = net(test_patch_1)
                    y2 = net(test_patch_2)
                    y3 = net(test_patch_3)
                    y = torch.softmax(y, dim=1)[0, :, :, :, :]
                    # y1 = torch.softmax(y1, dim=1)
                # y111 = y.cpu().data.numpy()#[0, :, :, :, :]
                # y1 = rot_back_process(y1,3)
                # y2 = rot_back_process(y2,2)
                # y3 = rot_back_process(y3,1)
                y1 = flip_back_process(y1,0)
                y2 = flip_back_process(y2,1)
                y3 = flip_back_process(y3,2)
                average = (y1+y2+y3+y)/4
                loss_here0 = criterion(average ,y).item()
                loss_here1 = criterion(average ,y1).item()
                loss_here2 = criterion(average ,y2).item()
                loss_here3 = criterion(average ,y3).item()
                loss_this_part = loss_here3+loss_here2+loss_here1 + loss_here0
                loss_this_data += loss_this_part
    cut_patch_number =  sx * sy * sz
    # print('fds')
    return  loss_this_data/cut_patch_number


def rot_back_process(tensor,axis = 0):
    tensor = tensor.cpu().data.numpy()[0, :, :, :, :]
    tensor_0,tensor_1 = tensor[0, :, :, :], tensor[1, :, :, :]
    tensor_0_back,tensor_1_back = np.rot90(tensor_0, axis),np.rot90(tensor_1, axis)
    tensor_back = np.concatenate([np.expand_dims(tensor_0_back,0),np.expand_dims(tensor_1_back,0)],0)
    tensor_back = torch.softmax(torch.from_numpy(tensor_back), dim=0).cuda()
    return tensor_back
def flip_back_process(tensor,axis = 0):
    if axis !=2:
        tensor = tensor.cpu().data.numpy()[0, :, :, :, :]
        tensor_0,tensor_1 = tensor[0, :, :, :], tensor[1, :, :, :]
        tensor_0_back,tensor_1_back = np.flip(tensor_0, axis),np.flip(tensor_1, axis)
        tensor_back = np.concatenate([np.expand_dims(tensor_0_back,0),np.expand_dims(tensor_1_back,0)],0)
        tensor_back = torch.softmax(torch.from_numpy(tensor_back), dim=0).cuda()
    else:
        tensor = tensor.cpu().data.numpy()[0, :, :, :, :]
        tensor_0, tensor_1 = tensor[0, :, :, :], tensor[1, :, :, :]
        tensor_0_back, tensor_1_back = np.flip(np.flip(tensor_0, 1),0), np.flip(np.flip(tensor_1, 1),0)
        tensor_back = np.concatenate([np.expand_dims(tensor_0_back, 0), np.expand_dims(tensor_1_back, 0)], 0)
        tensor_back = torch.softmax(torch.from_numpy(tensor_back), dim=0).cuda()
    return tensor_back


def test_all_case(net,  test_list=None, num_classes=4, patch_size=(96, 96, 96), stride_xy=64, stride_z=64):
    image_list = test_list
    # print("Validation begin")
    name_list = []
    consis_loss_list = []
    number=0
    for image_path in image_list:
        number+=1
        sys.stdout.write('\r%d/%s' % (number, len(image_list)))
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        loss_consis = test_single_case_for_consis_loss(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        name_list.append(image_path.split('/')[-1].split('.')[0])
        consis_loss_list.append(loss_consis)
    # print("Validation end")
    df = pd.DataFrame()
    # name_list_new = remove_list_img_h5(name_list)
    df['image_name'] = name_list
    df['loss'] = consis_loss_list
    df.to_csv('result.csv', index=False)

    df = pd.read_csv('result.csv')
    df = df.sort_values('loss', ascending=False)  # False: sort the consistency loss from big to small
    df.to_csv('result.csv', index=False)


test_list = pd.read_csv(os.path.join(os.getcwd(), args.imgs_infer_list))['image_name'].tolist()
data_path = 'dateset/BraTS2019/data/'
test_list = [''.join([data_path, '/', i + '.h5']) for i in test_list]

test_all_case(net,  test_list=test_list, num_classes=2, patch_size=(96, 96, 96), stride_xy=64, stride_z=64)

