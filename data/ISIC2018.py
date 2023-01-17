import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
# from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset

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

class ISIC(Dataset):
    def __init__(self, OOD_Condition = 'normal',Level = 0,dataset_folder='/ISIC2018_Task1_npy_all/',
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
                self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
                self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]
            # self.folder = sorted([join(dataset_folder, self.train_type, 'image', x) for x in
            #                       listdir(join(dataset_folder, self.train_type, 'image'))])
            # self.mask = sorted([join(dataset_folder, self.train_type, 'label', x) for x in
            #                     listdir(join(dataset_folder, self.train_type, 'label'))])
        else:
            with open(join(self.folder_file, self.train_type + '_subject.txt'),
                          'r') as f:
                self.image_list = f.readlines()
                self.image_list = [item.replace('\n', '') for item in self.image_list]
                self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
                self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in
                             self.image_list]
                self.folder_file = dataset_folder + folder.split('/')[0] + '/'
            if OOD_Condition == 'noise':
                with open(join(self.folder_file, self.train_type + '_subject'+'_N'+str(Level)+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder, 'image', x) for x in self.image_list]

            elif OOD_Condition == 'mask':
                with open(join(self.folder_file, self.train_type + '_subject'+'_PM'+str(Level)+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            elif OOD_Condition == 'blur':
                with open(join(self.folder_file, self.train_type + '_subject'+'_B'+str(Level)+'.txt'),
                          'r') as f:
                    self.image_list = f.readlines()
                    self.image_list = [item.replace('\n', '') for item in self.image_list]
                    self.condition_folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            else:
                self.condition_folder = self.folder
                print("Choosing condititon type error, You have to choose the loading data type including: normal, noise, mask, blur")

        assert len(self.folder) == len(self.mask)
        # if self.train_type =='train':
        #     del self.image_list[0:1813]
        #     del self.folder[0:1813]
        #     del self.mask[0:1813]

    def __getitem__(self, item: int):
        image = np.load(self.folder[item])
        label = np.load(self.mask[item])
        if self.train_type in ['train', 'val']:
            sample = {'image': image, 'label': label}
            if self.transform is not None:
                # TODO: transformation to argument datasets
                sample = self.transform(sample, self.train_type)
            return sample['image'], sample['label']
        else:
            condition_image = np.load(self.condition_folder[item])
            sample = {'image': image,'condition_image':condition_image, 'label': label}
            if self.transform is not None:
                # TODO: transformation to argument datasets
                sample = self.transform(sample, self.train_type)
            return sample['image'],sample['condition_image'], sample['label']



    def __len__(self):
        return len(self.folder)

# a = ISIC2018_dataset()
