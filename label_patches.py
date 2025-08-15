import os
from glob import glob
import matplotlib.pyplot as plt
import cv2

source_path3 = sorted(glob('/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/train/*_3.png'))
source_path6 = sorted(glob('/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/train/*_6.png'))
source_path = sorted(source_path3 + source_path6)

for img in source_path:
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    dx, dy = 64, 64
    grid_color = 0
    im[0:h:dy] = grid_color
    im[:, 0:w:dx] = grid_color

    plt.imshow(im, cmap='gray')
    plt.title(os.path.basename(img))
    plt.show()