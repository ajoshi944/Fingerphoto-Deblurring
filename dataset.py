import torch
from torchvision import transforms, datasets
from glob import glob
from PIL import Image
import os
import numpy as np
# define the image transforms and augmentations
train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, )) # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, )) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

train_transform2 = transforms.Compose([
    transforms.Grayscale(1),
])

class ImageScore(torch.utils.data.Dataset):
    def __init__(self, img_dir, score_file_path, transform=None, test=False):
        self.img_dir = img_dir
        self.list = glob(self.img_dir)
        self.score_file_path = score_file_path
        self.score_dict = {}
        self.transform = transform
        self.test = test
        f = open(self.score_file_path, 'r')
        for record in f:
            filename, score = record.split(' ')
            self.score_dict[filename.rstrip('.wsq')] = np.int(score.rstrip('\n'))

    def __getitem__(self, index):
        image_path = self.list[index]
        image_name = os.path.basename(image_path)
        score = self.score_dict[image_name.split('.')[0]]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            return image, image_name, score
        else:
            return image, score

    def __len__(self):
        return len(self.list)

class ImageScore2(torch.utils.data.Dataset):
    def __init__(self, img_dir, score_file_path, transform=None, test=False):
        self.img_dir = img_dir
        self.list = glob(self.img_dir)
        self.score_file_path = score_file_path
        self.score_dict = {}
        self.transform = transform
        self.test = test
        f = open(self.score_file_path, 'r')
        for record in f:
            record_list = record.split(' ')
            self.score_dict[record_list[0].rstrip('.wsq')] = (np.int(record_list[1].rstrip('\n')), [np.int(score.rstrip('\n')) for score in record_list[2:]])

    def __getitem__(self, index):
        image_path = self.list[index]
        image_name = os.path.basename(image_path)
        score, patch_score = self.score_dict[image_name.split('.')[0]]
        patch_score = np.asarray(patch_score)
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        # image = image.repeat([1,3,1,1])

        if self.test:
            return (image, image_name, score, patch_score)
        else:
            return image, score, patch_score

    def __len__(self):
        return len(self.list)

def get_datasets(clf=False, train=False, splits=False, batch_size=300):
    if clf:
        ############################### FOR CLASSIFICATION ##################################
        train_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/train/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_w_cl3_splits_scores.txt',
                                   transform=train_transform)

        val_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/val/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_val_w_cl3_splits_scores.txt',
                                 transform=val_transform)

        test_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final/test/*.png',
                                  '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_test_w_cl3_splits_scores.txt',
                                  transform=val_transform, test=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=66, shuffle=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        if train:
            return train_dataset, train_dataloader, val_dataset, val_dataloader
        else:
            return test_dataset, test_dataloader

    if splits:
        train_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/train/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_w_splits_scores.txt',
                                   transform=train_transform, test=False)

        val_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/val/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_val_w_splits_scores.txt',
                                 transform=val_transform)

        train_dataset2 = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/train/*.png',
                                    '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_w_splits_scores.txt',
                                    transform=train_transform2, test=False)

        val_dataset2 = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/val/*.png',
                                  '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_val_w_splits_scores.txt',
                                  transform=train_transform2)

        test_dataset = ImageScore2('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final/test/*.png', #  /home/n-lab/Amol/trial_codes/citer_poster/5615_tag_1603568_03042015_7_001_CrossMatchVerifier300LC_ac99.png
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_test_w_splits_scores.txt',#'/media/n-lab/Samsung-2TB/data/biocop_nist_final_ext/patch_test.txt',
                                  transform=val_transform, test=True)

        # test_dataset = ImageScore('/media/n-lab/Samsung-2TB/data/contactless/for_quality/*.png',
        #                         '/media/n-lab/Samsung-2TB/data/contactless/for_quality/nfiq_cl_trial.txt',
        #                         transform=val_transform, test=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=66, shuffle=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        if train:
            return train_dataset2, train_dataloader, val_dataset2, val_dataloader
        else:
            return test_dataset, test_dataloader

    else:
        # traning and validation datasets and dataloaders
        ############################## FOR REGRESSION ##################################
        train_dataset = ImageScore('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/train/*.png', #'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_ext/train/*.png',
                                    '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_scores.txt',#'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_ext/nfiq_biocop_nist_train_score_final.txt',
                                   transform=train_transform, test=False)

        train_dataset2 = ImageScore('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/train/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_scores.txt',
                                   transform=train_transform, test=False)

        val_dataset = ImageScore('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/val/*.png',#'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_ext/val/*.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_val_scores.txt',#'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_ext/nfiq_biocop_nist_val_score.txt'
                                 transform=val_transform)

        test_dataset = ImageScore('/home/n-lab/Amol/contact-quality2/data/biocop_nist_final/test/*.png', # /home/n-lab/Amol/contact-quality2/data/biocop_nist_final/for_poster/*.png  /home/n-lab/Amol/contact-quality2/data/biocop_nist_final/citer_poster/5615_tag_1603568_03042015_7_001_CrossMatchVerifier300LC_ac99.png',
                                   '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final/nfiq_biocop_nist_test_score.txt', #  nfiq_for_poster_score.txt
                                  transform=val_transform, test=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=66, shuffle=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        if train:
            return train_dataset, train_dataloader, val_dataset, val_dataloader
        else:
            return test_dataset, test_dataloader