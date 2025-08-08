import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from cv2 import imread
import cv2
import torchvision.transforms as transforms
from motion_blur.blur_image import BlurImage
from motion_blur.generate_trajectory import Trajectory
from motion_blur.generate_PSF import PSF
import math
import torch
random.seed(1)

transform_list = []

transform_list.append(transforms.Lambda(lambda img: __blur(img)))

A_transform = transforms.Compose(transform_list)


def crop_center(img,cropx=256,cropy=256): # set the cropx and cropy parameters to the image size.
    y, x = (296, 296)
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def __blur(img):
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]

    img = np.array(img)
    padded = np.pad(img, ((20, 20), (20, 20), (0, 0)), mode='edge')

    trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
    psf = PSF(canvas=64, trajectory=trajectory).fit()
    blurred_img = BlurImage(padded, PSFs=psf, part=np.random.choice([1, 2, 3])).blur_image()
    blurred_img = crop_center(blurred_img)

    # img = img.astype('float') / 255.
    blurred_img = blurred_img.astype('float') / 255.
    img = Image.fromarray(blurred_img)

    return img
