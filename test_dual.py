import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
# import models_mlp_binary_posencoding as models_mlp
# import models_mlp_backup as models_mlp
import resnet18_dual_w_SA as models_mlp
import argparse
from dataset_dual import get_datasets
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
import torchsummary
import numpy as np
import pickle
# matplotlib.style.use('ggplot')

"""
Command for training:
python train.py --exp_name vgg_5ksize_v5 --lr-scheduler --early-stopping 
"""
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
parser.add_argument('--exp_name', required=True)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
torch.autograd.set_detect_anomaly(True)

test_dataset, test_dataloader, dataset_name = get_datasets(clf=False, train=False, weights=False, batch_size=1)
exp_name = args.exp_name
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'model'
# instantiate the model
"""
Change the method to get different models.
exp_resnet18(): Complete resnet18 architecture with kernel size = 5 and # of kernels 16->32->64->128
get_vgg(): VGG11_bn architecture with kernel size = 5 and # of kernels 16->32->64->128
"""
model = models_mlp.updated_resnet18()
quality_model = models_mlp.Quality()
# fusing_model = models_mlp.Fuse()

#####################################
def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def tensor2np_img(t):
    """
    convert image from pytorch tensor to numpy array
    """
    ti_np = t.cpu().detach().numpy().squeeze()
    ti_np = norm_minmax(ti_np)
    if len(ti_np.shape) > 2:
        ti_np = ti_np.transpose(1, 2, 0)
    return ti_np


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = tensor2np_img(t[i])
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.axis('off')
    plt.show()
#####################################
def refine(s):
    if s < 0:
        return 0
    if s > 100:
        return 100
    return s

def compute_acc(l1, l2):
    corrects = 0
    corrects = np.sum(l1==l2)
    acc=100. * corrects/len(l1)
    return acc

model = model.cuda()
quality_model = quality_model.cuda()
checkpoint = torch.load(f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/{exp_name}/latest_model.pth')
model.load_state_dict(checkpoint['model'])
quality_model.load_state_dict(checkpoint['quality_model'])

# fusing_model = fusing_model.cuda()
# sharpness_score = open(f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/sharpness_{dataset_name}_padded_scores.txt', 'r')
# sharpness_dict = {}
# for record in sharpness_score:
#     filename, score = record.strip().split(' ')
#     sharpness_dict[filename] = int(score)
# model.module.load_state_dict(checkpoint['model'], strict=False)
f = open(f'./results/{exp_name}/scores_{dataset_name}_cc.txt', 'w')
# validation function
def validate(model, test_dataloader, val_dataset):
    print('Testing')
    model.eval()
    quality_model.eval()
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    transfer = lambda x: x.float().to(device)
    # load regressor model
    # regressor = pickle.load(open(f'./results/{exp_name}/lin_regressor.save', 'rb'))
    with torch.no_grad():
        for i, (candidate, filename, cand_nfiq) in prog_bar:
            counter += 1

            candidate = transfer(candidate)

            # target = transfer(target)
            # matching_score = transfer(matching_score)
            # cand_nfiq = transfer(cand_nfiq)

            cand_nfiq = cand_nfiq.cpu().numpy()
            # total += target.size(0)

            candidate_feats = model(candidate)
            quality_score, qm = quality_model(candidate_feats)

            # if i>6:
            #     exit()
            # candidate_feats = candidate_feats.squeeze().cpu().unsqueeze(0).numpy()
            quality_score = quality_score.squeeze(1).cpu().numpy()
            # quality_score = int((int(quality_score[0]) + sharpness_dict[str(filename[0])])*0.5)
            quality_score = int(quality_score[0])
            # if quality_score > 60:
            #     print(filename, quality_score)
            #     plot_tensor([candidate[0], qm[0]])
            # quality_score = regressor.predict(candidate_feats)[0]
            # quality_score = int(refine(quality_score))
            # print(filename[0], cand_nfiq[0], quality_score[0])
            # exit()
            # reference_feats = model(reference)
            f.write(f'{str(filename[0])} {cand_nfiq[0]} {quality_score}\n')

    f.close()

pathlib.Path(f'./results/{exp_name}').mkdir(parents=True, exist_ok=True)
start = time.time()
validate(model, test_dataloader, test_dataset)
end = time.time()
print(f"Testing time: {(end-start)/60:.3f} minutes for {len(test_dataset)} images.")
print('Saving results...')

print('TESTING COMPLETE')