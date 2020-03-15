#
# Ref:https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/custom_transforms.py
#

import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms.functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __init__(self, radius=(0., 1.)):
        self.radius = radius

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if np.random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(*self.radius)))

        return {'image': img, 'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class FixedResize(object):
    def __init__(self, size, is_resize=True):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)  # size: (h, w)
        self.is_resize = is_resize

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        if self.is_resize:
            mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'label': mask}


class AdjustColor(object):
    def __init__(self, factor=(0.3, 2.)):
        self.factor = factor

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        brightness_factor = np.random.uniform(*self.factor)
        contrast_factor = np.random.uniform(*self.factor)
        saturation_factor = np.random.uniform(*self.factor)

        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)

        return {'image': img, 'label': mask}


class CutOut(object):
    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        image = np.array(img)
        mask = np.array(mask)

        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]

        # find mask center coordinate
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)

        # left-top point
        xmin, ymin = cx - mask_size_half, cy - mask_size_half
        # right-bottom point
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size

        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

        if random.uniform(0, 1) < 0.5:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return {'image': Image.fromarray(image), 'label': Image.fromarray(mask)}


class RandomScale(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        image = np.array(image)
        mask = np.array(mask)

        scale = np.random.uniform(0.7, 1.5)
        h, w = image.shape[:2]
        aug_image = image.copy()
        aug_mask = mask.copy()

        aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)))
        aug_mask = cv2.resize(aug_mask, (int(scale * w), int(scale * h)))

        if scale < 1.:
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant", constant_values=0)
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant", constant_values=255)

        if scale >= 1.:
            new_h, new_w = aug_image.shape[:2]
            pre_h_crop = int((new_h - h) / 2)
            pre_w_crop = int((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]

        return {'image': Image.fromarray(aug_image), 'label': Image.fromarray(aug_mask)}


class Translate(object):
    def __init__(self, t=50, ingore_index=255):
        self.t = t
        self.ingore_index = ingore_index

    def __call__(self, sample):
        image = sample['image']
        target = sample['label']
        image = np.array(image)
        target = np.array(target)

        if np.random.random() > 0.5:
            x = random.uniform(-self.t, self.t)
            y = random.uniform(-self.t, self.t)
            M = np.float32([[1, 0, x],
                            [0, 1, y]])
            h, w = image.shape[:2]
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            target = cv2.warpAffine(target, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(self.ingore_index, ))

        return {'image': Image.fromarray(image), 'label': Image.fromarray(target)}

