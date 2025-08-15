import copy
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.transforms import functional as tF
from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple

RECT_NAMEDTUPLE = namedtuple('RECT_NAMEDTUPLE', 'x1 x2 y1 y2')


class RandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, resample=None, expand=False, center=None, fill=None):
        super(RandomRotation, self).__init__(degrees=degrees, resample=resample, expand=expand, center=center,
                                             fill=fill)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        angle = self.get_params(self.degrees)
        return tF.rotate(img, angle, self.resample, self.expand, self.center, self.fill), angle


# class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
#     def __init__(self, p):
#         super(RandomHorizontalFlip, self).__init__(p)
#
#     def forward(self, img):  # override forward
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be flipped.
#         Returns:
#             PIL Image or Tensor: Randomly flipped image.
#         """
#
#         if torch.rand(1) < self.p:
#             return tF.hflip(img), 1
#         return img, 0


class RandomColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.8):
        super(RandomColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = tF.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = tF.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = tF.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = tF.adjust_hue(img, hue_factor)

        return img


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img):

        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return tF.hflip(img), 1
        return img, 0


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super().__init__(size=size, scale=scale, ratio=ratio)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        #
        # i = 8
        # j = 8
        # h = 16
        # w = 16

        return tF.resized_crop(img, i, j, h, w, self.size, self.interpolation), i, j, h, w

    def forward2(self, img, i, j, h, w):
        return tF.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class CustomDataset(Dataset):
    def __init__(self, dataset, img_w=32, deg_max=45, scale_min=0.2, mean=None, std=None, normalizer=None):
        self.dataset = dataset
        self.totensor = transforms.ToTensor()
        self.img_w = img_w
        self.deg_max = deg_max
        self.t1 = RandomResizedCrop(self.img_w, scale=(scale_min, 1))
        self.random_hflip = RandomHorizontalFlip()



        self.t_rest = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

        self.t_rest2 = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.h_rotate = RandomRotation(deg_max)

    def __getitem__(self, index):
        img1_orig, label, patch_score = self.dataset[index]

        # a = np.array(img1)

        img1, _ = self.random_hflip(img1_orig)




        # img1 = np.array(img1)
        padd_sz = self.img_w//5


        if self.deg_max>0:
            img1 = tF.pad(img1, [padd_sz]*4, padding_mode='reflect')
            img1, _ = self.h_rotate(img1)
            img1 = tF.center_crop(img1, self.img_w)

        # flip1 = 0
        # flip2 = 0
        # if flip1 == flip2:
        #     flip = 0
        # else:
        #     flip = 1

        i, j, h, w = self.t1.get_params(img1, self.t1.scale, self.t1.ratio)

        # i = 2
        # j = 5
        # w = 10
        # h = 10
        # i = 8
        # j = 8
        # i = 2 # distance from the top
        # j = 12 # distance from the left
        # h = 20 # vertical size
        # w = 10 # horizontal size
        #
        # i = 8
        # j = 2
        # h = 16
        # w = 26

        pos0 = j + int(np.round(w / 2))  # np.random.randint(j, j + w + 1, size=[1])
        pos1 = i + int(np.round(h / 2))  # np.random.randint(i, i + h + 1, size=[1])
        pos = np.array([pos0, pos1])

        # add marker on image for debugging
        debug = False
        if debug:

            img1 = np.array(img1, dtype=np.uint8)  # np.asarray(img1, dtype=np.uint8)
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if pos1 + a >= 0 and pos1 + a < self.img_w and pos0 + b >= 0 and pos0 + b < self.img_w:
                        img1[pos1 + a, pos0 + b] = 0

            img1 = Image.fromarray(img1)

        img2 = copy.deepcopy(img1)

        # img = resize(img, size, interpolation)

        img2 = self.t1.forward2(img2, i, j, h, w)

        # scale_x = self.img_w / w
        # scale_y = self.img_w / h

        # apply t1

        # img1 = self.t1(img1)
        # img2 = self.t1(img2)

        # rotate images
        img2 = tF.pad(img2, [padd_sz] * 4, padding_mode='reflect')
        img2, deg2 = self.h_rotate(img2)
        img2 = tF.center_crop(img2, self.img_w)

        img2, flip2 = self.random_hflip(img2)

        img1 = self.t_rest(img1)
        img2 = self.t_rest(img2)

        img_orig = self.t_rest2(img1_orig)
        return (img1, img2, -deg2, flip2, i, j, h, w, pos, label, patch_score, img_orig)

    def __len__(self):
        return len(self.dataset)
