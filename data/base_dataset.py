"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
# from cv2 import imread
import cv2
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from motion_blur.blur_image import BlurImage
from motion_blur.generate_trajectory import Trajectory
from motion_blur.generate_PSF import PSF
import math
import torch
random.seed(1)
import torchvision.transforms.functional as TF

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.8

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, blur=False, isTrain=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'squarepad' in opt.preprocess:
        transform_list.append(SquarePad())
        transform_list.append(transforms.Resize((256, 256), method))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if blur:
        # APPLY MOTION BLUR KERNEL
        mask = get_mask()
        b = random.choice([0, 1])# 0:Gaussian, 1:Motion
        blurring_type = b
        blurring_type = torch.as_tensor(blurring_type)
        transform_list.append(transforms.Lambda(lambda img: __blur(img, isTrain, mask, b)))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    if blur:
        return transforms.Compose(transform_list), 1-mask, blurring_type
    return transforms.Compose(transform_list)


def get_mask():
    gaussian = random.choice([True, False])
    #gaussian = True
    img_size = 256

    mask = torch.zeros([img_size, img_size])

    if gaussian:
        kernel_size = np.random.randint(img_size // 4 * 2, img_size)

        ratio = np.random.randint(5, 7)

        sigma = kernel_size / ratio

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
        # Make sure sum of values in gaussian kernel equals 1.
        # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()

        x = np.random.randint(0, img_size - kernel_size)
        y = np.random.randint(0, img_size - kernel_size)
        mask[x:x + kernel_size, y:y + kernel_size] = gaussian_kernel

    mask = mask.repeat(3, 1, 1)

    return mask


class SquarePad:
    def __call__(self, image):
        s = image.size
        max_wh = max(s[-1], s[-2])
        hp = int((max_wh - s[-2]) / 2)
        vp = int((max_wh - s[-1]) / 2)
        padding = [hp, vp, hp, vp]
        return TF.pad(image, padding, 0, 'constant')

def crop_center(img,cropx=256,cropy=256):
    y, x = (296, 296)
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def __blur(img, isTrain, mask, b):
    #b = random.choice([0, 1]) # 0:Gaussian, 1:Motion
    #b = 1
    if isTrain:
        sigma = round(random.uniform(1.5, 4.0), 1)
    else:
        sigma = 3.0
    k = int((6 * sigma) - 1)
    if k % 2 == 0:
        k += 1
    mask_np = mask.numpy().squeeze()
    mask_3 = mask_np.transpose(1, 2, 0)
    if b == 0:
        open_cv_image = np.array(img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        blurred_img = cv2.GaussianBlur(open_cv_image, (k, k), sigma)
        img = open_cv_image.astype('float') / 255.
        blurred_img = blurred_img.astype('float') / 255.
        out = blurred_img * (1 - mask_3) + img * mask_3
        out = (out * 255).astype('uint8')
        img = Image.fromarray(out)
    elif b == 1:
        if isTrain:
            params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
        else:
            params = [0.009]

        img = np.array(img)
        padded = np.pad(img, ((20, 20), (20, 20), (0, 0)), mode='edge')

        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        blurred_img = BlurImage(padded, PSFs=psf, part=np.random.choice([1, 2, 3])).blur_image()
        blurred_img = crop_center(blurred_img)

        img = img.astype('float') / 255.
        blurred_img = blurred_img.astype('float') / 255.
        out = blurred_img * (1 - mask_3) + img * mask_3
        out = (out * 255).astype('uint8')
        img = Image.fromarray(out)

    return img

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
