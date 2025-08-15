from PIL import Image
from PIL import ImageFilter
from glob import glob
import os
import numpy as np
import random
import cv2
source = '/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/local_score_aug_test/*.png'
target = '/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/local_score_aug_test/'
for img in glob(source):
    img_name = os.path.basename(img)
    im = Image.open(img)
    rot_img = im.rotate(angle=25.)
    flip_img = im.transpose(Image.FLIP_LEFT_RIGHT)

    coord = random.randint(75, 181)
    mask = np.ones((256, 256), dtype=np.uint8)
    mask = cv2.circle(mask, (coord, coord), 64, (0, 0, 0), -1)
    bimg = np.array(im)
    blurred_img = cv2.GaussianBlur(bimg, (5, 5), 3)
    blurred_img = blurred_img.astype('float') / 255.
    bimg = bimg.astype('float') / 255.
    blurred_img = blurred_img * (1 - mask) + bimg * (mask)
    blur_img = Image.fromarray((blurred_img*255.).astype('uint8'))

    rot_img.save(f'{target}{img_name[:5]}rot.png', dpi=(500, 500))
    flip_img.save(f'{target}{img_name[:5]}flip.png', dpi=(500, 500))
    blur_img.save(f'{target}{img_name[:5]}blur.png', dpi=(500, 500))