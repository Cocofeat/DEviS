import os
import PIL
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from os import listdir
from os.path import join
from PIL import Image
# from utils.transform import itensity_normalize
# from data.transform import LiTS2017_transform
from torch.utils.data.dataset import Dataset
import argparse


def itensity_normalize(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized n                                                                                                                                                                 d volume
    """

    # pixels = volume[volume > 0]
    mean = volume.mean()
    std = volume.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]

    return out

class DRIVE(Dataset):
    def __init__(self, OOD_Condition='normal', Level=0, dataset_folder='/DRIVE/',
                 folder='folder0', train_type='train', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.folder_file = dataset_folder + folder
        self.folder = []
        self.condition_folder = []
        if self.train_type in ['train', 'val']:
            # this is for cross validation
            with open(join(self.folder_file, self.train_type + '_subject.txt'),
                      'r') as f:
                self.image_list = f.readlines()
                self.image_list = [item.replace('\n', '') for item in self.image_list]
                self.folder = [join(dataset_folder, 'Image', x) for x in self.image_list]
                self.mask = [join(dataset_folder, 'Mask',  x.split('_')[0]+'_manual1.png' ) for x in self.image_list]
            # self.folder = sorted([join(dataset_folder, self.train_type, 'image', x) for x in
            #                       listdir(join(dataset_folder, self.train_type, 'image'))])
            # self.mask = sorted([join(dataset_folder, self.train_type, 'label', x) for x in
            #                     listdir(join(dataset_folder, self.train_type, 'label'))])
        else:
            with open(join(self.folder_file, self.train_type + '_subject.txt'),
                      'r') as f:
                self.image_list = f.readlines()
                self.image_list = [item.replace('\n', '') for item in self.image_list]
                self.folder = [join(dataset_folder, 'Image', x) for x in self.image_list]
                self.mask = [join(dataset_folder, 'Mask',  x) for x in self.image_list]
                self.folder_file = dataset_folder + folder.split('/')[0] + '/'
            if OOD_Condition == 'IC':
                with open(join(self.folder_file, self.train_type + '_subject'+'_IC'+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder,  'Image', x) for x in self.image_list]
            elif OOD_Condition == 'B':
                with open(join(self.folder_file, self.train_type + '_subject'+'_B'+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder, 'Image', x) for x in self.image_list]
            elif OOD_Condition == 'ICB':
                with open(join(self.folder_file, self.train_type + '_subject'+'_ICB'+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder,  'Image', x) for x in self.image_list]
            elif OOD_Condition == 'ICBL':
                with open(join(self.folder_file, self.train_type + '_subject'+'_ICBL'+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder,  'Image', x) for x in self.image_list]
            else:
                self.condition_folder = self.folder
                print("Choosing condititon type error, You have to choose the loading data type including: normal, noise, mask, blur")

        assert len(self.folder) == len(self.mask)
        # if self.train_type =='train':
        #     del self.image_list[0:1813]
        #     del self.folder[0:1813]
        #     del self.mask[0:1813]

    def __getitem__(self, item: int):
        imgitk = sitk.ReadImage(self.folder[item])
        image = sitk.GetArrayFromImage(imgitk)
        image = image.astype(np.float32)
        # image = image.astype(np.float32) / 255.0
        # labelitk = sitk.ReadImage(self.mask[item])
        # label = sitk.GetArrayFromImage(labelitk)
        labelitk = Image.open(self.mask[item])
        label = np.array(labelitk)
        # m = np.unique(label)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        if self.train_type in ['train', 'val']:
            sample = {'image': image, 'label': label}
            if self.transform is not None:
                # TODO: transformation to augument datasets
                sample = self.transform(sample, self.train_type)
            # plt.figure(3)
            # plt.imshow(image[7,:,:])
            # plt.show()
            # plt.figure(4)
            # plt.imshow(label[7,:,:])
            # plt.show()
            # plt.figure(5)
            # plt.imshow(sample['image'].squeeze(0).numpy()[7,:,:])
            # plt.show()
            # plt.figure(6)
            # plt.imshow(sample['label'].numpy()[7,:,:])
            # plt.show()
            return sample['image'], sample['label']
        else:
            condition_imgitk = sitk.ReadImage(self.condition_folder[item])
            condition_image = sitk.GetArrayFromImage(condition_imgitk)
            # condition_image = condition_image.astype(np.float32) / 255.0
            sample = {'image': image, 'condition_image': condition_image, 'label': label}
            if self.transform is not None:
                # TODO: transformation to augument datasets
                sample = self.transform(sample, self.train_type)
            return sample['image'], sample['condition_image'], sample['label']

    def __len__(self):
        return len(self.folder)


if __name__ == '__main__':
    # a = ISIC2018_dataset()
    parser = argparse.ArgumentParser()
    # training detalis
    parser.add_argument('--end_epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--submission', default='./results', type=str)

    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate')  # BraTS: 0.0002 # ISIC: 0.0002
    # DataSet Information
    parser.add_argument('--savepath', default='./results/plot/output', type=str)
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument("--mode", default="train&test", type=str, help="train/test/train&test")
    parser.add_argument('--dataset', default='DRIVE', type=str, help="BraTS/ISIC/COVID/CHAOS/LiTS")  #
    parser.add_argument("--folder", default="folder5", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument('--batch_size', default=16, type=int, help="2/4/8/16")
    parser.add_argument('--OOD_Condition', default='noise', type=str, help="normal/noise/mask/blur/spike/ghost/")
    parser.add_argument('--OOD_Level', default=1, type=int,
                        help="0: 'No',1:'Low', 2:'Upper Low', 3:'Mid', 4:'Upper Mid', 5:'High'")
    # parser.add_argument('--OOD_Variance', default=2, type=int)
    parser.add_argument('--snapshot', default=True, type=bool, help="True/False")  # visualization results
    parser.add_argument('--Uncertainty_Loss', default=True, type=bool, help="True/False")  # adding uncertainty_loss
    parser.add_argument('--input_modality', default='four', type=str, help="t1/t2/both/four")  # Single/multi-modal
    parser.add_argument('--model_name', default='U', type=str, help="U/V/AU/TransU/ViT/")  # multi-modal
    parser.add_argument('--test_epoch', type=int, default=197, metavar='N',
                        help='best epoch')
    args = parser.parse_args()
    root_path = 'E:/Coco_file/LiTS/'
    exp_folder = args.folder
    trainset = DRIVE(dataset_folder=root_path, train_type='train', folder=exp_folder,
                    transform=LiTS2017_transform)
    validset = DRIVE(dataset_folder=root_path, train_type='val', folder=exp_folder,
                    transform=LiTS2017_transform)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True, pin_memory=True)
    testset = DRIVE(OOD_Condition=args.OOD_Condition, Level=args.OOD_Level, dataset_folder=root_path, train_type='test',
                   folder=exp_folder,
                   transform=LiTS2017_transform)
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)

    for i, data in enumerate(train_loader):
        x, target = data
        if args.mode == 'test':
            noise = torch.clamp(torch.randn_like(x) * args.Variance, -args.Variance * 2, args.Variance * 2)
            x += noise
        # x_no = np.unique(x.numpy())
        # target_no = np.unique(target.numpy())