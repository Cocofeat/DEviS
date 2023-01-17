import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pickle
from scipy import ndimage
import argparse
import matplotlib.pyplot as plt

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}

class Random_intencity_shiftboth(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}

class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)

class Pad_condition(object):
    def __call__(self, condition_image):
        image = condition_image
        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        return image

class Padboth(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155,n)>(240,240,160,n)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

class ToTensorboth(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

class ToTensor_condition(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, condition_image):
        image = condition_image
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float().permute([3, 0, 1, 2])

        return image

def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)

def transform_condition(condiiton_image):
    trans = transforms.Compose([
        Pad_condition(),
        # MaxMinNormalization(),
        ToTensor_condition()
    ])

    return trans(condiiton_image)

def transformboth(sample):
    trans = transforms.Compose([
        Padboth(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shiftboth(),
        ToTensorboth()
    ])

    return trans(sample)


def transformboth_valid(sample):
    trans = transforms.Compose([
        Padboth(),
        # MaxMinNormalization(),
        ToTensorboth()
    ])

    return trans(sample)

class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train', modal='t1',OOD_Condition = 'normal',folder='folder0',level = 0):
        self.lines = []
        paths, names,only_paths = [], [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                only_root = root.split('/')[0]+'/'+root.split('/')[1]+'/'+root.split('/')[2]
                only_path = os.path.join(only_root, line, name + '_')
                only_paths.append(only_path)
                self.lines.append(line)
        self.mode = mode
        del paths[1:35]
        del only_paths[1:35]
        del names[1:35]
        del self.lines[1:35]
        self.modal = modal
        self.names = names
        self.image_list = names
        self.paths = paths
        self.only_paths = only_paths
        self.OOD_Condition = OOD_Condition
        self.folder = folder
        self.level = level

    def __getitem__(self, item):
        path = self.paths[item]
        only_path = self.only_paths[item]
        if self.mode in ['train']:
            image, label = pkload(only_path + 'data_f32b04M.pkl')
            # print(np.unique(label))
            label[label==4]=3
            # print(np.unique(label))
            if self.modal == 'two':
                sample = {'image': image[..., 0:1] , 'label': label}
                sample = transform(sample)
            else:
                sample = {'image': image, 'label': label}
                sample = transformboth(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(only_path + 'data_f32b04M.pkl')
            label[label == 4] = 3
            if self.modal == 't1':
                sample = {'image': image[..., 0], 'label': label}
                sample = transform_valid(sample)
            elif self.modal =='t2':
                sample = {'image': image[..., 1], 'label': label}
                sample = transform_valid(sample)
            else:
                sample = {'image': image, 'label': label}
                sample = transformboth_valid(sample)
            return sample['image'], sample['label']
        else:
            image,label = pkload(only_path + 'data_f32b04M.pkl')
            label[label == 4] = 3
            if self.modal == 'two':
                sample = {'image': image[..., 0:1], 'label': label}
                sample = transform_valid(sample)
                return sample['image'], sample['label']
            else:
                sample = {'image': image, 'label': label}

                if self.OOD_Condition == 'noise':
                    condition_image = pkload(only_path + 'data_f32b04M' + '_' + self.folder + '_N'+str(self.level)+ '.pkl')
                elif self.OOD_Condition == 'blur':
                    condition_image = pkload(only_path + 'data_f32b04M' + '_' + self.folder + '_B'+str(self.level) + '.pkl')
                elif self.OOD_Condition == 'mask':
                    condition_image = pkload(only_path + 'data_f32b04M' + '_' + self.folder + '_PM'+str(self.level) + '.pkl')
                elif self.OOD_Condition == 'ghost':
                    condition_image = pkload(only_path + 'data_f32b04M' + '_' + self.folder + '_GH'+str(self.level) + '.pkl')
                elif self.OOD_Condition == 'spike':
                    condition_image = pkload(only_path + 'data_f32b04M' + '_' + self.folder + '_SP'+str(self.level) + '.pkl')
                else:
                    condition_image = None
                    print(
                        "Choosing condititon type error, You have to choose the loading data type including: normal, noise, mask, blur")
                # plt.figure(1)  # 图像窗口名称
                # plt.imshow(image[:,:,70,1,])
                # plt.axis('on')  # 关掉坐标轴为off
                # plt.title('text title')  # 图像标题
                # plt.show()
                # plt.figure(2)  # 图像窗口名称
                # plt.imshow(condition_image[:,:,70,1,])
                # plt.axis('on')  # 关掉坐标轴为off
                # plt.title('text title')  # 图像标题
                # plt.show()
                # plt.figure(3)  # 图像窗口名称
                # plt.imshow(label[:,:,70,])
                # plt.axis('on')  # 关掉坐标轴为off
                # plt.title('text title')  # 图像标题
                # plt.show()
                sample = transformboth_valid(sample)
                condition_image = transform_condition(condition_image)
                sample['condition_image']= condition_image
                return sample['image'],sample['condition_image'], sample['label']

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='C:/Coco_file/BraTSdata/archive2019', type=str)
    parser.add_argument('--train_dir', default='MICCAI_BraTS_2019_Data_TTraining', type=str)
    parser.add_argument('--valid_dir', default='MICCAI_BraTS_2019_Data_TValidation', type=str)
    parser.add_argument('--test_dir', default='MICCAI_BraTS_2019_Data_TTest', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--train_file', default='D:/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Ttrain_subject.txt', type=str)
    parser.add_argument('--valid_file', default='D:/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Tval_subject.txt', type=str)
    parser.add_argument('--test_file', default='D:/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/Ttest_subject.txt', type=str)
    parser.add_argument('--dataset', default='brats', type=str)
    parser.add_argument('--num_gpu', default= 4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--modal', default='both', type=str)
    parser.add_argument('--Variance', default=0.1, type=int)
    args = parser.parse_args()
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    val_root = os.path.join(args.root, args.valid_dir)
    test_list = os.path.join(args.root, args.test_dir, args.test_file)
    test_root = os.path.join(args.root, args.test_dir)
    train_set = BraTS(train_list, train_root, args.mode,args.modal)
    val_set = BraTS(val_list, val_root, args.mode, args.modal)
    test_set = BraTS(test_list, test_root, args.mode, args.modal)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    # train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // args.num_gpu,
    #                           drop_last=True, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    for i, data in enumerate(train_loader):
        x, target = data
        if args.mode == 'test':
            noise = torch.clamp(torch.randn_like(x) * args.Variance, -args.Variance * 2, args.Variance * 2)
            x += noise
        # x_no = np.unique(x.numpy())
        # target_no = np.unique(target.numpy())