import torch
import random
import PIL
import numbers
import numpy as np
import torch.nn as nn
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as ts
from scipy import ndimage
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import math

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class Random_rotate_3d(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Random_Flip_3d(object):
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

class random_scale_rotate_translate_3d(object):
    # implemented with affine transformation
    def __call__(self, sample, scale = [0.3, 0.3, 0.3], rotate = [30, 30, 30], translate = [0, 0, 0], noshear = True
):
        image = sample['image']
        label = sample['label']

        tensor_img = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor_lab = torch.from_numpy(label).long().unsqueeze(0).unsqueeze(0)


        if isinstance(scale, float) or isinstance(scale, int):
            scale = [scale] * 3
        if isinstance(translate, float) or isinstance(translate, int):
            translate = [translate] * 3
        if isinstance(rotate, float) or isinstance(rotate, int):
            rotate = [rotate] * 3

        scale_z = 1 - scale[0] + np.random.random() * 2 * scale[0]
        scale_x = 1 - scale[1] + np.random.random() * 2 * scale[1]
        scale_y = 1 - scale[2] + np.random.random() * 2 * scale[2]
        shear_xz = 0 if noshear else np.random.random() * 2 * scale[0] - scale[0]
        shear_yz = 0 if noshear else np.random.random() * 2 * scale[0] - scale[0]
        shear_zx = 0 if noshear else np.random.random() * 2 * scale[1] - scale[1]
        shear_yx = 0 if noshear else np.random.random() * 2 * scale[1] - scale[1]
        shear_zy = 0 if noshear else np.random.random() * 2 * scale[2] - scale[2]
        shear_xy = 0 if noshear else np.random.random() * 2 * scale[2] - scale[2]
        translate_z = np.random.random() * 2 * translate[0] - translate[0]
        translate_x = np.random.random() * 2 * translate[1] - translate[1]
        translate_y = np.random.random() * 2 * translate[2] - translate[2]

        theta_scale = torch.tensor([[scale_y, shear_xy, shear_zy, translate_y],
                                [shear_yx, scale_x, shear_zx, translate_x],
                                [shear_yz, shear_xz, scale_z, translate_z],
                                [0, 0, 0, 1]]).float()
        angle_xy = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi
        angle_xz = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
        angle_yz = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi

        theta_rotate_xz = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(angle_xz), -math.sin(angle_xz), 0],
                                    [0, math.sin(angle_xz), math.cos(angle_xz), 0],
                                    [0, 0, 0, 1]]).float()
        theta_rotate_xy = torch.tensor([[math.cos(angle_xy), -math.sin(angle_xy), 0, 0],
                                    [math.sin(angle_xy), math.cos(angle_xy), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).float()
        theta_rotate_yz = torch.tensor([[math.cos(angle_yz), 0, -math.sin(angle_yz), 0],
                                    [0, 1, 0, 0],
                                    [math.sin(angle_yz), 0, math.cos(angle_yz), 0],
                                    [0, 0, 0, 1]]).float()

        theta = torch.mm(theta_rotate_xy, theta_rotate_xz)
        theta = torch.mm(theta, theta_rotate_yz)
        theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)

        grid = F.affine_grid(theta, tensor_img.size(), align_corners=True)
        tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros',
                               align_corners=True).long()

        image,label =  tensor_img.squeeze(0).squeeze(0).numpy(), tensor_lab.squeeze(0).squeeze(0).numpy()
        # plt.figure(1)
        # plt.imshow(sample['image'][7, :, :])
        # plt.show()
        # plt.figure(2)
        # plt.imshow(sample['label'][7, :, :])
        # plt.show()
        return {'image': image, 'label': label}

class Random_intencity_shift_3d(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[1, image.shape[1], 1])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1])

        image = image * scale_factor + shift_factor

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        # plt.figure(1)
        # plt.imshow(sample['image'][7, :, :])
        # plt.show()
        # plt.figure(2)
        # plt.imshow(sample['label'][7, :, :])
        # plt.show()

        label = np.ascontiguousarray(label)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

def lits_transform(sample):
    trans = ts.Compose([
        # random_scale_rotate_translate_3d(),
        # Random_rotate_3d(),  # time-consuming
        Random_Flip_3d(),
        # Random_intencity_shift_3d(),
        ToTensor()
    ])

    return trans(sample)
    
def LiTS2017_transform(sample, train_type):
    # image, label = Image.fromarray(sample['image']),Image.fromarray(sample['label'])
    image, label= sample['image'], sample['label']

    if train_type in ['train', 'val']:
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        # image = ts.ToTensor()(image)
        if np.random.random() < 0.2:
            sample = lits_transform(sample)
            image, label= sample['image'], sample['label']
        else:
            image = torch.from_numpy(image).float().unsqueeze(0)
            label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}
    else:
        condition_image = sample['condition_image']
        # image = ts.ToTensor()(image)
        # image = ts.ToTensor()(image)
        # condition_image = ts.ToTensor()(condition_image)
        image = torch.from_numpy(image).float().unsqueeze(0)
        condition_image = torch.from_numpy(condition_image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {'image': image, 'condition_image':condition_image, 'label': label}
def CHAOS2020_transform(sample, train_type):
    # image, label = Image.fromarray(sample['image']),Image.fromarray(sample['label'])
    image, label = sample['image'], sample['label']
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(label)
    # plt.show()
    if train_type in ['train', 'val']:
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        # image = ts.ToTensor()(image)
        image =  ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, ), std=(0.5,))])(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}
    else:
        condition_image = sample['condition_image']
        # image = ts.ToTensor()(image)
        # image = ts.ToTensor()(image)
        # condition_image = ts.ToTensor()(condition_image)
        image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, ), std=(0.5,))])(image)
        condition_image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, ), std=(0.5,))])(condition_image)
        label = ts.ToTensor()(label)

        return {'image': image, 'condition_image':condition_image, 'label': label}

def ISIC2018_transform(sample, train_type):
    image, label = Image.fromarray(np.uint8(sample['image']), mode='RGB'), \
                   Image.fromarray(np.uint8(sample['label']), mode='L')
    if train_type in ['train', 'val']:
        image, label = randomcrop(size=(224, 300))(image, label)
        image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}
    else:
        condition_image = Image.fromarray(np.uint8(sample['condition_image']), mode='RGB')
        image,label = resize(size=(224, 300))(image, label)
        condition_image,condition_image = resize(size=(224, 300))(condition_image,condition_image)
        image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
        condition_image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(condition_image)
        label = ts.ToTensor()(label)

        return {'image': image, 'condition_image':condition_image, 'label': label}

def DRIVE2004_transform(sample, train_type):
    image, label = Image.fromarray(np.uint8(sample['image']), mode='RGB'), \
                   Image.fromarray(np.uint8(sample['label']))
    label = label.convert("L")
    if train_type in ['train', 'val']:
        # image, label = randomcrop(size=(224, 300))(image, label)
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}
    else:
        condition_image = Image.fromarray(np.uint8(sample['condition_image']), mode='RGB')
        image,label = resize(size=(584, 565))(image, label)
        condition_image,condition_image = resize(size=(584, 565))(condition_image,condition_image)
        image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
        condition_image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(condition_image)
        label = ts.ToTensor()(label)

        return {'image': image, 'condition_image':condition_image, 'label': label}

def TOAR2019_transform(sample, train_type):
    image, label = sample['image'],sample['label']
    # label = label.convert("L")
    if train_type in ['train', 'val']:
        # image, label = randomcrop(size=(224, 300))(image, label)
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        # image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(image)
        image = ts.Compose([ts.ToTensor()])(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}
    else:
        condition_image = np.uint8(sample['condition_image'])
        # image,label = resize(size=(512, 512))(image, label)
        # condition_image,condition_image = resize(size=(512, 512))(condition_image,condition_image)
        # image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(image)
        # condition_image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(condition_image)
        image = ts.Compose([ts.ToTensor()])(image)
        condition_image = ts.Compose([ts.ToTensor()])(condition_image)
        label = ts.ToTensor()(label)

        return {'image': image, 'condition_image':condition_image, 'label': label}

def HC_2018_transform(sample, train_type):
    image, label = sample['image'],sample['label']
    # label = label.convert("L")
    if train_type in ['train', 'val']:
        # image, label = randomcrop(size=(224, 300))(image, label)
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
        # image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(image)
        image = ts.Compose([ts.ToTensor()])(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}
    else:
        condition_image = np.uint8(sample['condition_image'])
        condition_image = Image.fromarray(condition_image)
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        image,label = resize(size=(128, 1024))(image, label)
        condition_image,condition_image = resize(size=(128, 1024))(condition_image,condition_image)
        # image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(image)
        # condition_image = ts.Compose([ts.ToTensor(),
        #                 ts.Normalize(mean=(0.5), std=(0.5))])(condition_image)
        image = ts.Compose([ts.ToTensor()])(image)
        condition_image = ts.Compose([ts.ToTensor()])(condition_image)
        label = ts.ToTensor()(label)

        return {'image': image, 'condition_image':condition_image, 'label': label}

def COVID2019_transform(sample, train_type):
    # image, label = Image.fromarray(np.uint8(sample['image']), mode='RGB'),\
    #                Image.fromarray(np.uint8(sample['label']), mode='L')
    #
    # if train_type == 'train':
    #     # image, label = randomcrop(size=(224, 300))(image, label)
    #     image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
    # # else:
    # #     image, label = resize(size=(224, 300))(image, label)
    #
    # image = ts.Compose([ts.ToTensor(),
    #                     ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
    # label = ts.ToTensor()(label)
    #
    # return {'image': image, 'label': label}
    if train_type in ['train', 'val']:
        image, condition_image,label = sample['image'],sample['condition_image'],sample['label']
        image = ts.ToTensor()(image)
        condition_image = ts.ToTensor()(condition_image)
        label = ts.ToTensor()(label)
        return {'image': image, 'condition_image':condition_image, 'label': label}
    else:
        image, label = sample['image'],sample['label']
        image = ts.ToTensor()(image)
        label = ts.ToTensor()(label)
        return {'image': image, 'label': label}

# these are founctional function for transform
def randomflip_rotate(img, lab, p=0.5, degrees=0):
    if random.random() < p:
        img = TF.hflip(img)
        lab = TF.hflip(lab)
    if random.random() < p:
        img = TF.vflip(img)
        lab = TF.vflip(lab)

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    angle = random.uniform(degrees[0], degrees[1])
    img = TF.rotate(img, angle)
    lab = TF.rotate(lab, angle)

    return img, lab


        
class randomcrop(object):
    """Crop the given PIL Image and mask at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lab (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image and mask.
        """
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            lab = TF.pad(lab, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = TF.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            lab = TF.pad(lab, (int((1 + self.size[1] - lab.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = TF.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            lab = TF.pad(lab, (0, int((1 + self.size[0] - lab.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), TF.crop(lab, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class resize(object):
    """Resize the input PIL Image and mask to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lab (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image and mask.
        """
        return TF.resize(img, self.size, self.interpolation), TF.resize(lab, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


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