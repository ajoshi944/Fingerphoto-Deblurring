import pathlib
import operator

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
# import models_mlp_binary_posencoding as models_mlp
# import models_mlp_backup as models_mlp
import resnet18_baseline_w_SA as models_mlp
import argparse
from sklearn import metrics
# from dataset_binary_cl import get_datasets
from dataset_dual import get_datasets
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
import torchsummary
import numpy as np
# matplotlib.style.use('ggplot')
from matplotlib.colors import ListedColormap
"""
Command for training:
python test_binary_cl.py --exp_name vgg_5ksize_v5
"""

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
parser.add_argument('--exp_name', required=True)
parser.add_argument('--local', dest='local', action='store_true')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


test_dataset, test_dataloader, dataset_name = get_datasets(clf=False, train=False, weights=False, batch_size=1)
exp_name = args.exp_name
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'model'

if args.lr_scheduler:
     # change the accuracy, loss plot names and model.txt name
    model_name = 'lrs_model'

# instantiate the model
model = models_mlp.updated_resnet18()
# model = models.cnn_model()


#####################################
def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())

# def plot_tensor(t, varname):
#     """
#     plot pytorch tensors
#     input: list of tensors t
#     """
#     # rdylgn = matplotlib.colormaps['viridis'].resampled(8)
#     for i in range(len(t)):
#         ti_np = tensor2np_img(t[i])
#         if ti_np.shape[0] < 6:
#             ti_np = cv2.resize(ti_np, (32, 32), interpolation=cv2.INTER_CUBIC)
#             plt.subplot(1, len(t), i + 1)
#             plt.imshow(ti_np, cmap=matplotlib.cm.get_cmap('RdYlGn')) #
#             plt.axis('off')
#         else:
#             plt.subplot(1, len(t), i + 1)
#             plt.imshow(ti_np, cmap='gray')
#             plt.axis('off')
#     plt.title(varname)
#     plt.show()

def tensor2np_img(t, normalize=True):
    """
    convert image from pytorch tensor to numpy array
    """
    ti_np = t.cpu().detach().numpy().squeeze()
    if normalize:
        ti_np = norm_minmax(ti_np)
    if len(ti_np.shape) > 2:
        ti_np = ti_np.transpose(1, 2, 0)
    return ti_np


def plot_tensor(t, scores=None, filename='Unk'):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    fig, ax = plt.subplots(1, 2)
    viridis_respampled = plt.cm.get_cmap('RdYlGn', 10)
    for i, tensor in enumerate(t):
        if i==0:
            ti_np = tensor2np_img(tensor)
            ax.ravel()[i].imshow(ti_np, cmap='gray')
            ax.ravel()[i].set_axis_off()
        elif i==1:
            ti_np = tensor2np_img(tensor, normalize=True)
            print(ti_np)
            ax.ravel()[i].imshow(ti_np, cmap=viridis_respampled, vmin=0.0, vmax=1.)#
            ax.ravel()[i].set_axis_off()
        if scores:
            ax.ravel()[i].set_title(scores[i].cpu().detach().numpy())
    # fig.delaxes(ax[3][-3])
    # fig.delaxes(ax[3][-2])
    # fig.delaxes(ax[3][-1])
    plt.tight_layout()
    plt.axis('off')
    fig.suptitle(filename)
    # for i in range(len(t)):
    #     ti_np = tensor2np_img(t[i])
    #     plt.subplot(1, len(t), i + 1).set_title(scores[i])
    #     plt.imshow(ti_np, cmap='gray')
    plt.show()

def compute_acc(l1, l2):
    corrects = 0
    corrects = np.sum(l1==l2)
    acc=100. * corrects/len(l1)


    return acc

def plot(f, is_local, logits=False):
    diff_list = []
    if is_local:
        acc_list = []
        roc_list = []
    if is_local:
        patches = np.arange(start=1, stop=18, dtype=int)
        true_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                     14: [], 15: [], 16: [], 17: []}
        pred_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                     14: [], 15: [], 16: [], 17: []}
    else:
        patches = np.arange(start=1, stop=2, dtype=int)
        true_dict = {1: []}
        pred_dict = {1: []}
    for record in f:
        record_list = record.strip('\n').split(' ')
        for patch in patches:
            true_dict[patch].append(float(record_list[(patch * 2) - 1]))
            if logits:
                pred_dict[patch].append(0.0 if float(record_list[patch * 2]) < 0.5 else 1.0)
            else:
                pred_dict[patch].append(float(record_list[patch * 2]))


    if is_local:
        for (tk, true_values), (pk, pred_values) in zip(true_dict.items(), pred_dict.items()):
            if tk == 1 and pk == 1:
                full_acc = compute_acc(true_values, pred_values)
                full_fpr, full_tpr, _ = metrics.roc_curve(true_values, pred_values)
            else:
                acc_list.append(compute_acc(np.array(true_values), np.array(pred_values)))
                roc_list.append(metrics.roc_curve(np.array(true_values), np.array(pred_values)))

        # diff = np.asarray(true, dtype=int) - np.asarray(pred, dtype=int)
        # mean_list = [np.abs(diff).mean() for diff in diff_list]
        print(full_acc)
        print(acc_list)
        # wanted_keys = range(2, 18)
        # for (_, true_classes), (_, pred_classes) in zip(true_dict[wanted_keys], pred_dict[wanted_keys]):
        #     print(len(true_classes), len(pred_classes))
        #     exit()
    else:
        for (_, true_values), (_, pred_values) in zip(true_dict.items(), pred_dict.items()):
            diff_list.append([compute_acc(true_values, pred_values)])
        # diff = np.asarray(true, dtype=int) - np.asarray(pred, dtype=int)
        # mean_list = [np.abs(diff).mean() for diff in diff_list]
        # print(mean_list)
    # mu, std = norm.fit(diff)
    if is_local:
        fig, axes = plt.subplots(4, 5, figsize=(15, 15))
        fig.delaxes(axes[3][-3])
        fig.delaxes(axes[3][-2])
        fig.delaxes(axes[3][-1])
        mod_ax = [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[0, 4], axes[1, 0], axes[1, 1], axes[1, 2],
                  axes[1, 3], axes[1, 4], axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3], axes[2, 4], axes[3, 0],
                  axes[3, 1]]
    else:
        fig, axes = plt.subplots(1, 1)
        mod_ax = [axes]

    if is_local:
        # for i, diff in enumerate(diff_list):
            # sns.histplot(diff, kde=False, color='red', ax=mod_ax[i], legend=False).set(
            #     title=f'{"Global" if i == 0 else f"Patch {i}"}' + f' mean error: {round(mean_list[i], 4)}',
            #     xlabel='Range', ylabel='Count')  # xlim=(-40,40)
        # plt.plot(diff_list, axes=mod_ax[0])
        full_auc = metrics.auc(full_fpr, full_tpr)
        display = metrics.RocCurveDisplay(fpr=full_fpr, tpr=full_tpr, roc_auc=full_auc, estimator_name='Full image')
        display.plot(ax=mod_ax[0])

        for i, roc in enumerate(roc_list, start=1):
            auc = metrics.auc(roc[0], roc[1])
            display = metrics.RocCurveDisplay(fpr=roc[0], tpr=roc[1], roc_auc=auc, estimator_name=f'Patch {i}')
            display.plot(ax=mod_ax[i])
        plt.show()
    else:
        for i, diff in enumerate(diff_list):
            if i == 0:
                c = "blue"
            else:
                c = "red"
            sns.histplot(diff, kde=False, color=c, ax=mod_ax[i], legend=False).set(
                title=f'{"Global" if i == 0 else f"Patch {i}"}' + f' mean error: {round(diff_list[i], 4)}',
                xlabel='Range', ylabel='Count')  # xlim=(-40,40)


    fig.tight_layout()


    # plt.suptitle('Score error distribution')
    # plt.savefig(f"./results/{exp_name}/error_plot.png")
    # plt.show()
    # # accuracy plots
    # plt.figure(figsize=(10, 7))
    # plt.plot(test_accuracy, color='blue', label='Test accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig(f"./results/{exp_name}/{acc_plot_name}.png")
    # plt.show()
#####################################

model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
model = model.module
checkpoint = torch.load(f"./results/{exp_name}/1_{model_name}.pth")
model.load_state_dict(checkpoint['model'])
f = open(f'./results/{exp_name}/scores_{dataset_name}.txt', 'w')
# validation function
def validate(model, test_dataloader, test_dataset, is_local):
    print('Validating')
    model.eval()
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset)/test_dataloader.batch_size))
    # f = open(f'./results/{exp_name}/scores.txt', 'w')

    for i, (candidate, filename, cand_nfiq) in prog_bar:
        with torch.no_grad():
            counter += 1
            #data, target = data[0].to(device), data[1].to(device)
            img = candidate.float().to(device)
            target = cand_nfiq.float().to(device)
            # filename = data[1]
            # if is_local:
            #     patch_class = data[3].float().to(device)


            total += target.size(0)
            outputs, classes = model(img, test=True)
            if is_local:
                patch_class = patch_class.squeeze(0)
            classes = classes.squeeze(0)

            outputs = outputs.squeeze(0)

            if is_local:
                pred = round(outputs[0].item(), 0)
                filename = ''.join(filename)
                if 30.0 <= pred <= 40.0:
                    print(f'Ground truth: {patch_class},\n Predicted: {classes}')
                    plot_tensor([img, classes.view(4, 4), patch_class.view(4, 4)],None,
                            f'{filename}\nGround truth:{np.round(target[0].item(), 0)} Pred:{pred}')
            # if ''.join(filename) in [  # '5029341_10232014_7_001_UPEKEikonTouch700_ac2.png',
                #                 '54461_tag_7052091_10282015_5_002_CrossMatchVerifier300LC_ac0.png',
                #                 '57289_tag_7855831_09162015_4_001_CrossMatchVerifier300LC_ac0.png',
                #                 '2550791_05182015_2_002_UPEKEikonTouch700_ac0.png',
                #                 '50630_tag_6007713_01262015_4_002_CrossMatchVerifier300LC_ac5.png',
                #                 '2_tag_8500381_09022011_9_2_CrossMatchVerifier300LC_ac5.png',
                #                 '812_tag_1511965_10282014_7_001_CrossMatchVerifier300LC_ac99.png',
                #                 '2137864_03192015_1_001_UPEKEikonTouch700_ac99.png',
                #                 '00002420_R_500_slap_01_2_ac99.png'
                # '5379_tag_4073772_04162015_1_002_CrossMatchVerifier300LC_ac4.png',
                # '3636713_04142015_7_002_UPEKEikonTouch700_ac3.png',
                # '1498_tag_2100061_03162015_3_002_CrossMatchVerifier300LC_ac99.png',
                # '322_tag_1603568_03042015_10_001_CrossMatchVerifier300LC_ac99.png',
                # '1367794_07102015_6_001_UPEKEikonTouch700_ac99.png',
                # '2905144_11092011_5_UPEKEikonTouch700_2_ac3.png',
                # '812_tag_1511965_10282014_7_001_CrossMatchVerifier300LC_ac99.png',
                # '2137864_03192015_1_001_UPEKEikonTouch700_ac99.png',
                # '00002420_R_500_slap_01_2_ac99.png'
                # '8991451_02162015_3_001_UPEKEikonTouch700_ac99.png',
                # '00002401_J_500_palm_10_2_ac99.png',
                # '339_tag_2100061_07102015_6_002_CrossMatchVerifier300LC_ac99.png',
                # '1821598_11122014_5_001_UPEKEikonTouch700_ac99.png',
                # '1561421_05082015_10_001_UPEKEikonTouch700_ac99.png',
            # 'c27_1.png', 'c402_1.png', 'c544_6.png', 'c134_1.png', 'c465_1.png', 'c669_6.png', 'c97_1.png',
            #     'c436_1.png', 'c731_6.png', 'c731_6_bl.png','c731_6_bl2.png', 'c359_3.png', 'c359_3_bl.png',
            # 'c662_2.png', 'c662_2_bl.png'
            # 'c638_1.png', 'c638_2.png', 'c638_3.png', 'c638_4.png', 'c638_5.png', 'c638_6.png',
            # 'c676_1.png', 'c676_2.png', 'c676_3.png', 'c676_4.png', 'c676_5.png', 'c676_6.png',
            # 'c701_1.png', 'c701_2.png', 'c701_3.png', 'c701_4.png', 'c701_5.png', 'c701_6.png',
            # 'c794_1.png', 'c794_2.png', 'c794_3.png', 'c794_4.png', 'c794_5.png', 'c794_6.png']:
            # if is_local:
            #     #     print(f'Ground truth: {patch_class}\n Predicted: {classes}')
            #     #     plot_tensor([img, patch_class.view(4, 4), classes.view(4, 4)],
            #     #             f'{filename}\nGround truth:{np.round(target[0].item(), 2)} Pred:{round(outputs[0].item(), 2)}')
            #     # else:
            #     plot_tensor([img, classes.view(4, 4)],
            #                 filename=f'{filename}\nNFIQ:{target[0].item()} Pred:{round(outputs[0].item(), 0)}')
                # exit()
    #         if is_local:
    #             f.write(f'{str(filename[0])} {round(outputs[0].item(), 0)} '
    #                     f'{round(classes[0].item(), 0)} {round(classes[1].item(), 0)} '
    #                     f'{round(classes[2].item(), 0)} {round(classes[3].item(), 0)} '
    #                     f'{round(classes[4].item(), 0)} {round(classes[5].item(), 0)} '
    #                     f'{round(classes[6].item(), 0)} {round(classes[7].item(), 0)} '
    #                     f'{round(classes[8].item(), 0)} {round(classes[9].item(), 0)} '
    #                     f'{round(classes[10].item(), 0)} {round(classes[11].item(), 0)} '
    #                     f'{round(classes[12].item(), 0)} {round(classes[13].item(), 0)} '
    #                     f'{round(classes[14].item(), 0)} {round(classes[15].item(), 0)}'
    #                     f'\n')
    #         else:
            else:
                f.write(f'{str(filename[0])} {cand_nfiq[0]} {int(outputs[0].item())}\n')
    #     # if counter == 3000:
    #     #     break
    # f.close()


start = time.time()
validate(model, test_dataloader, test_dataset, is_local=args.local)
end = time.time()
print(f"Testing time: {(end-start)/60:.3f} minutes")


print('Saving results...')
pathlib.Path(f'./results/{exp_name}').mkdir(parents=True, exist_ok=True)



# f = open(f'./results/{exp_name}/scores_cl_test.txt', 'r')
# plot(f, is_local=args['local'], logits=True)

print('TESTING COMPLETE')