import torch
import random
from torchvision import transforms, datasets
from glob import glob
from PIL import Image
import os
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms.functional as TF
import cv2
class SquarePad:
    def __call__(self, image):
        s = image.size
        max_wh = max(s[-1], s[-2])
        hp = int((max_wh - s[-2]) / 2)
        vp = int((max_wh - s[-1]) / 2)
        padding = [hp, vp, hp, vp]
        return TF.pad(image, padding, 0, 'constant')

# define the image transforms and augmentations
train_transform = transforms.Compose([
        transforms.Grayscale(1),
        SquarePad(),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((256, 256), antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, )) # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

val_transform = transforms.Compose([
        transforms.Grayscale(1),
        SquarePad(),
        transforms.Resize((256, 256), antialias=None),
        # transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, )) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

qm_transform = transforms.Compose([
        # transforms.Grayscale(1),
        SquarePad(),
        transforms.Resize((64, 64), antialias=None, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, ), std=(0.5, )) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

train_transform2 = transforms.Compose([
    transforms.Grayscale(1),
])
def get_sharpness_score(image, lower_threshold=45, upper_threshold=180):
    height, width = image.shape
    edges = cv2.Canny(image, lower_threshold, upper_threshold)

    # Save the result
    # cv2.imwrite(output_path, edges)
    # display_images(image, masked_image, edges)
    edges = edges / 255.
    edges_sum = np.sum(edges)
    total_size = height * width  # 256*256#(np.pi*((width//3)-1)*191) - (np.pi*31*47)
    return int((edges_sum / total_size) * 100.)

class ImageScore(torch.utils.data.Dataset):
    def __init__(self, img_dir, score_file_path, transform=None):
        self.img_dir = img_dir
       # self.list = glob(self.img_dir)
        self.score_file_path = score_file_path
        self.score_dict = {}
        self.transform = transform
        self.qm_dir = '/home/n-lab/Amol/fingerphoto_quality/all_data_qm/'
        # self.qm_list = os.listdir(self.qm_dir)
        f = open(self.score_file_path, 'r')
        for record in f:
            candidate_name, reference_name, matching_score, label, cand_nfiq, ref_nfiq, target, cand_innq, ref_innq, cand_sh, ref_sh, cand_ivniq, ref_ivniq = record.strip().split(' ')
            self.score_dict[(candidate_name, reference_name)] = (float(matching_score), int(label), int(cand_nfiq), int(ref_nfiq), int(target), int(cand_innq), int(ref_innq), int(cand_sh), int(ref_sh), int(cand_ivniq), int(ref_ivniq))
        self.data_list = list(self.score_dict.keys())

    def __getitem__(self, index):
        key_pair = self.data_list[index]
        candidate_name, reference_name = key_pair
        matching_score, label, cand_nfiq, ref_nfiq, target, cand_innq, ref_innq, cand_sh, ref_sh, cand_ivniq, ref_ivniq = self.score_dict[key_pair]
        candidate_path = os.path.join(self.img_dir, candidate_name)
        reference_path = os.path.join(self.img_dir, reference_name)

        cand_qm_path = os.path.join(self.qm_dir, candidate_name)
        ref_qm_path = os.path.join(self.qm_dir, reference_name)

        candidate = Image.open(candidate_path)
        reference = Image.open(reference_path)

        candidate_qm = Image.open(cand_qm_path)
        reference_qm = Image.open(ref_qm_path)

        cqm = np.asarray(candidate_qm)
        sh_ratio = np.sum(cqm, where=cqm>=127) / (cqm.sum()+1e-6)

        if self.transform is not None:
            candidate = self.transform(candidate)
            reference = self.transform(reference)

        candidate_qm = qm_transform(candidate_qm)
        reference_qm = qm_transform(reference_qm)


        candidate = candidate.repeat([3, 1, 1])
        # candidate = torch.cat([candidate, candidate_qm], dim=0)
        reference = reference.repeat([3, 1, 1])
        # reference = torch.cat([reference, reference_qm], dim=0)

        return {'candidate': candidate, 'reference': reference, 'matching_score': matching_score, 'label': label,
                'cand_innq': cand_innq, 'ref_innq': ref_innq, 'target': target, 'candidate_qm': candidate_qm,
                'cand_nfiq': cand_nfiq, 'ref_nfiq': ref_nfiq, 'reference_qm': reference_qm,
                'candidate_sh': cand_sh, 'reference_sh': ref_sh, 'candidate_ivniq': cand_ivniq, 'ref_ivniq': ref_ivniq, 'sh_ratio': sh_ratio, 'candidate_name':candidate_name}

    def __len__(self):
        return len(self.data_list)

class ImageScore2(torch.utils.data.Dataset):
    def __init__(self, img_dir, score_file_path, transform=None, img_ext='.png'):
        self.img_dir = img_dir
        # self.list = glob(self.img_dir)
        self.img_ext = img_ext
        self.score_file_path = score_file_path
        self.score_dict = {}
        self.transform = transform
        f = open(self.score_file_path, 'r')
        for record in f:
            record_list = record.split(' ')
            code = int(record_list[0].split('_')[0][1:])
            if 537 <= code < 821:  # biocop_only
                score = int(record_list[1].rstrip('\n'))
                # patch_score = np.asarray([np.int(s.rstrip('\n')) for s in record_list[2:]])
                # self.score_dict[record_list[0].rstrip('.wsq')] = (1 if 0.38 < score/100. else 0, np.where(patch_score/100. > 0.38, 1, 0))
                # self.score_dict[record_list[0].replace('.wsq', self.img_ext)] = np.round(score/100., 2)#, np.round(patch_score/100., 2))
                self.score_dict[record_list[0].replace('.wsq', self.img_ext)] = score
    def __getitem__(self, index):
        image_name, score_tuple = list(self.score_dict.items())[index]#self.list[index]
        # image_name = os.path.basename(image_path)
        score = score_tuple#self.score_dict[image_name.split('.')[0]]
        # score = 1 if 0.5 < score/100. else 0
        # patch_score = [1 if 0.5 < s/100. else 0 for s in patch_score]
        # patch_score = np.asarray(patch_score)
        # patch_score = np.where(patch_score/100. > 0.5, 1, 0)
        image = Image.open(os.path.join(self.img_dir, image_name))

        if self.transform is not None:
            image = self.transform(image)
        image = image.repeat([3,1,1])

        # if self.test:
        return (image, image_name, score)
        # else:
        #     return image, score

    def __len__(self):
        return len(self.score_dict.keys())

class ImageScore3(torch.utils.data.Dataset):
    def __init__(self, img_dir, score_file_path, transform=None, test=False):
        self.img_dir = img_dir
        # self.list = glob(self.img_dir)
        self.score_file_path = score_file_path
        self.score_dict = {}
        self.tr = transform
        self.test = test
        f = open(self.score_file_path, 'r')
        for record in f:
            self.patch_flag = False
            record_list = record.strip().split(' ')
            img_name = record_list[0]
            score = np.int(record_list[1].rstrip('\n'))
            if len(record_list) > 2:
                patch_score = np.asarray([np.int(s.rstrip('\n')) for s in record_list[2:]])
                self.patch_flag = True
            else:
                patch_score = np.ones((1, 16), dtype=int)
            if self.test:
                self.score_dict[record_list[0]] = (score, patch_score)
            # else:
            #     if '_6.png' in img_name or '_3.png' in img_name:
            else:
                self.score_dict[record_list[0]] = (score, patch_score, self.patch_flag)

    def transform(self, image, patches=None):
        tf_grayscale = transforms.Grayscale(1)
        image = tf_grayscale(image)
        if patches is not None:
            image = TF.hflip(image)
            patches = patches.reshape((4, 4))
            patches = np.fliplr(patches)
            patches = patches.reshape((1, 16))
        else:
            if random.random() > 0.5:
                image = TF.hflip(image)
        tf_Tensor = transforms.ToTensor()
        image = tf_Tensor(image)

        tf_normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
        image = tf_normalize(image)

        return image, patches

    def __getitem__(self, index):
        image_name, score_tuple = list(self.score_dict.items())[index]
        image_name = image_name.replace('wsq', 'png')
        # image_name = os.path.basename(image_path)
        if self.test:
            score, patch_score = score_tuple# self.score_dict[image_name.split('.')[0]]
        else:
            score, patch_score, patch_flag = score_tuple
        # score = 1 if 0.5 < score/100. else 0
        # patch_score = [1 if 0.5 < s/100. else 0 for s in patch_score]
        # patch_score = np.asarray(patch_score)
        # patch_score = np.where(patch_score/100. > 0.5, 1, 0)
        image = Image.open(os.path.join(self.img_dir, image_name))

        if self.tr is not None:
            # image = self.transform(image)
            if self.test:
                image = self.tr(image)
            else:
                if patch_flag:
                    image, patch_score = self.transform(image, patch_score)
                else:
                    image, _ = self.transform(image)
        image = image.repeat([3,1,1])

        if self.test:
            return (image, image_name, score, patch_score)
        else:
            return {'candidate': image, 'target': score, }#'candidate_qm': candidate_qm,'cand_nfiq': cand_nfiq, 'ref_nfiq': ref_nfiq, 'reference_qm': reference_qm,'candidate_sh': cand_sh, 'reference_sh': ref_sh, 'candidate_ivniq': cand_ivniq, 'ref_ivniq': ref_ivniq}image, score, patch_score, patch_flag

    def __len__(self):
        return len(self.score_dict.keys())

def get_datasets(clf=False, train=False, batch_size=100, weights=False, ddp=False):
    splits = not clf
    # traning and validation datasets and dataloaders
    ############################## FOR REGRESSION ##################################
    # train_dataset = ImageScore3(
    #     '/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/transformed/',
    #     '/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/ecdf_stats/train_biocop_vin_mean_c10_w_manual_label.txt',
    #     '/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_train_w_splits_scores.txt',
        # transform=train_transform, test=False)

    train_dataset = ImageScore('/home/n-lab/Amol/fingerphoto_quality/all_set',
        '/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/ivniq_mean_equal_train_data_all_set.txt',#'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/in_train_data_all_set.txt',
        transform=train_transform)

    # val_dataset = ImageScore('/home/n-lab/Amol/fingerphoto_quality/all_set',
    #     '/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/ecdf_stats/val_polyu_in_mean_c5.txt',
    #     transform=val_transform)
    dataset_name = 'transformed'
    test_dataset = ImageScore2(f'/media/n-lab/DATA1/Amol/fingerphoto/test/{dataset_name}/',
                               f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/in_{dataset_name}_quality.txt',#f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/in_{dataset_name}_quality.txt',#
                               transform=val_transform)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=66, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if train:
        return train_dataset, train_dataloader#, val_dataset, val_dataloader
    else:
        return test_dataset, test_dataloader, dataset_name