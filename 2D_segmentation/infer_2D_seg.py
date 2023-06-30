import os
import torch
import time
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
from create_dataset import test_transform,Mydataset_infer
import argparse
import segmentation_models_pytorch as smp
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_infer_path', type=str, default=None,htlp='the img dir of the img you require to secelt for labeled' )
parser.add_argument('--weight_path', type=str, default='checkpoint/resnet34/fold0/fold0train01/ckpt.pth', help='checkpoint path acquired by labeled data')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='0'
begin_time = time.time()

test_transform = test_transform
name_list = os.listdir(args.imgs_infer_path)
infer_images2 = [''.join([args.imgs_infer_path,'/',i]) for i in name_list]
# infer_images2 is the image list of the all img path(absolute path for every image)
test_ds = Mydataset_infer(infer_images2, test_transform)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=False,pin_memory=False,num_workers=4,)

# rot = torchvision.transforms.functional.rotate
criterion = nn.MSELoss().to('cuda')
def main():
    model =smp.Unet(encoder_name='resnet34',encoder_weights=None,classes=2).to('cuda')
    model = model.to('cuda')
    model.load_state_dict(torch.load(args.weight_path))
    number=0
    model.eval()
    loss_list ,name_list= [],[]
    with torch.no_grad():
        for batch_idx,(name,imgs) in enumerate(test_dl):
            number+=1
            sys.stdout.write('\r%d/%s' % (number, len(test_dl)))
            name_pic = name[0].split('/')[-1]
            imgs = imgs.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            out1 = model(imgs.flip([-1])).flip([-1])
            out2 = model(imgs.flip([-2])).flip([-2])
            out3 = model(imgs.flip([-1, -2])).flip([-1, -2])
            # out4 = rot(model(rot(imgs,90).flip([-1])).flip([-1]),-90)
            # out5 = rot(model(rot(imgs,90).flip([-2])).flip([-2]),-90)
            # out6 = rot(model(rot(imgs,90).flip([-1,-2])).flip([-1,-2]),-90)
            # out7 = rot(model(rot(imgs,90)),-90)
            # print(masks_pred)
            masks_pred = torch.softmax(masks_pred,1)
            # print('masks_predsof',masks_pred)
            out1, out2, out3 = torch.softmax(out1,1),torch.softmax(out2,1),torch.softmax(out3,1)
                                                       # torch.softmax(out4,1),torch.softmax(out5,1),torch.softmax(out6,1),\
                                                       # torch.softmax(out7,1)
            average =  (out1 + out2 + out3 + masks_pred) / 4
            # mm = torch.cat((masks_pred,out1,out2,out3),0)
            loss = criterion(torch.cat((masks_pred,out1,out2,out3),0),
                             torch.cat((average,average,average,average),0)).item()
            loss_list.append(loss)
            name_list.append(name_pic)
        df = pd.DataFrame()
        df['image_name'] = name_list
        df['loss'] = loss_list
        df.to_csv('result.csv', index=False)

        df = pd.read_csv('result.csv')
        df = df.sort_values('loss',ascending=False)  #False: sort the consistency loss from big to small
        df.to_csv('result.csv', index=False)




if __name__ == '__main__':
    main()



