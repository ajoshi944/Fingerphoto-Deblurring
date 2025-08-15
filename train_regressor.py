import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
# import models_mlp_binary_posencoding as models_mlp
# import models_mlp_backup as models_mlp
import resnet18_baseline_w_CBAM as models_mlp
import argparse

from dataset_binary_cl import get_datasets
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
import torchsummary
import numpy as np

# matplotlib.style.use('ggplot')

"""
Command for training:
python train.py --exp_name vgg_5ksize_v5 --lr-scheduler --early-stopping 
"""

# train_score_counts = np.asarray([1372, 1007, 1149, 1202, 1136, 1142, 1237, 1140, 1179, 1115, 1143,
#                 1096, 1180, 1138, 1229, 1235, 1341, 1379, 1333, 1441, 1463,
#                 1577, 1632, 1730, 1909, 2033, 2154, 2176, 2429, 2489, 2635,
#                 2721, 2862, 3014, 3170, 3201, 3340, 3413, 3462, 3484, 3470,
#                 3588, 3721, 3683, 3775, 3722, 3755, 3783, 3796, 3785, 3865,
#                 3511, 3545, 3467, 3314, 3234, 3023, 2830, 2724, 2417, 2100,
#                 2092, 1862, 1614, 1442, 1203, 1002,  831,  718,  591,  467,
#                 364,  276,  219,  141,  132,   79,   51,   35,   15,   26,
#                 11, 4, 0.001, 3, 1, 0.001, 1, 0.001, 0.001, 0.001,
#                 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

# 600,000
# train_score_counts = np.asarray([1.4225e+04,1.5870e+04,1.7722e+04,1.7627e+04,1.6663e+04,1.6710e+04
# ,1.6310e+04,0.0000e+00,1.5682e+04,1.4397e+04,1.3863e+04,1.2933e+04
# ,1.2607e+04,1.2472e+04,1.2346e+04,0.0000e+00,1.2928e+04,1.3197e+04
# ,1.3557e+04,1.3999e+04,1.4275e+04,1.4335e+04,1.4206e+04,0.0000e+00
# ,1.3855e+04,1.3830e+04,1.3498e+04,1.3552e+04,1.3176e+04,1.3071e+04
# ,0.0000e+00,1.2801e+04,1.2797e+04,1.2316e+04,1.2108e+04,1.2010e+04
# ,1.1840e+04,1.1743e+04,0.0000e+00,1.1374e+04,1.0966e+04,1.0545e+04
# ,1.0075e+04,9.6720e+03,9.2990e+03,8.6300e+03,0.0000e+00,8.2820e+03
# ,7.7830e+03,7.2050e+03,6.7810e+03,6.2110e+03,5.7860e+03,0.0000e+00
# ,5.2560e+03,4.9300e+03,4.5140e+03,4.1920e+03,3.7840e+03,3.4160e+03
# ,3.1840e+03,0.0000e+00,2.8600e+03,2.6390e+03,2.3390e+03,2.1300e+03
# ,1.9470e+03,1.7320e+03,1.4260e+03,0.0000e+00,1.3730e+03,1.1350e+03
# ,9.7900e+02,9.3800e+02,7.4400e+02,6.2300e+02,0.0000e+00,5.0000e+02
# ,4.1300e+02,3.3600e+02,2.6600e+02,2.0400e+02,1.3800e+02,1.0900e+02
# ,0.0000e+00,7.9000e+01,6.5000e+01,4.7000e+01,2.5000e+01,2.1000e+01
# ,7.0000e+00,1.5000e+01,0.0000e+00,1.0000e+00,3.0000e+00,0.0000e+00
# ,2.0000e+00,1.0000e+00,0.0000e+00,1.0000e+00])

# 557,000
train_score_counts = np.asarray([8.5530e+03, 7.6710e+03, 8.0230e+03, 7.9360e+03, 7.6990e+03, 7.6090e+03
                                    , 7.6820e+03, 7.2230e+03, 6.8790e+03, 0.0000e+00, 6.5210e+03, 6.2150e+03
                                    , 5.9630e+03, 5.6660e+03, 5.6480e+03, 5.5610e+03, 5.6020e+03, 5.6480e+03
                                    , 0.0000e+00, 5.6580e+03, 5.7880e+03, 6.0610e+03, 6.3430e+03, 6.5390e+03
                                    , 7.0270e+03, 7.2020e+03, 7.8440e+03, 0.0000e+00, 8.1530e+03, 8.3900e+03
                                    , 9.0500e+03, 9.5560e+03, 9.7610e+03, 1.0310e+04, 1.0619e+04, 1.0835e+04
                                    , 0.0000e+00, 1.0853e+04, 1.1006e+04, 1.1159e+04, 1.0917e+04, 1.0947e+04
                                    , 1.0982e+04, 1.0812e+04, 1.0904e+04, 0.0000e+00, 1.0896e+04, 1.0700e+04
                                    , 1.0701e+04, 1.0646e+04, 1.0673e+04, 1.0569e+04, 1.0608e+04, 1.0670e+04
                                    , 0.0000e+00, 1.0475e+04, 1.0403e+04, 1.0136e+04, 9.8440e+03, 9.7430e+03
                                    , 9.1650e+03, 8.7380e+03, 8.5100e+03, 0.0000e+00, 7.8560e+03, 7.2540e+03
                                    , 6.7470e+03, 5.9790e+03, 5.3880e+03, 4.7520e+03, 4.2960e+03, 3.6830e+03
                                    , 0.0000e+00, 3.2460e+03, 2.6960e+03, 2.3310e+03, 1.9660e+03, 1.5050e+03
                                    , 1.2480e+03, 9.0900e+02, 7.5200e+02, 0.0000e+00, 5.7100e+02, 4.2600e+02
                                    , 2.7600e+02, 2.1700e+02, 1.3800e+02, 9.9000e+01, 5.7000e+01, 5.4000e+01
                                    , 0.0000e+00, 3.7000e+01, 1.6000e+01, 9.0000e+00, 8.0000e+00, 1.0000e+00
                                    , 4.0000e+00, 2.0000e+00, 0.0000e+00, 1.0000e+00])

min_count = 1200

train_score_counts = np.clip(train_score_counts, a_min=min_count, a_max=1000000000)

p = train_score_counts / train_score_counts.sum()
alpha_train = 100 / np.sum(1 / p)
w_train = alpha_train * (1 / p)
w_val = w_train  # alpha_val * (1 / p)
#
# print(w_train)
# exit()
#
#
# val_score_counts = np.asarray([19, 15, 15, 15,  6,  8, 12, 11, 15, 13,  7,
#                                9, 14,  9, 12, 10,  7, 10, 13, 12, 14,
#                                12, 12,  9, 15, 11, 16, 17, 17, 20, 17,
#                                20, 17, 19, 21, 27, 22, 29, 20, 39, 25,
#                                19, 28, 21, 21, 14, 22,  8, 24, 21, 13,
#                                13, 16, 10,  7,  8,  7, 13,  9,  9,  4,
#                                1, 0.001,  2,  1,  0.001, 0.001, 0.001, 1, 0.001, 1,
#                                0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.001, 0.001, 0.001, 0.001,
#                                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
#                                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
#
#
#
#
# p = val_score_counts / val_score_counts.sum()
# alpha_val = 100 / np.sum(1 / p)


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
parser.add_argument('--exp_name', required=True)
args = vars(parser.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
torch.autograd.set_detect_anomaly(True)

train_dataset, train_dataloader, val_dataset, val_dataloader = get_datasets(clf=False, train=True)

# instantiate the model
"""
Change the method to get different models.
exp_resnet18(): Complete resnet18 architecture with kernel size = 5 and # of kernels 16->32->64->128
get_vgg(): VGG11_bn architecture with kernel size = 5 and # of kernels 16->32->64->128
"""
# model = models_mlp.updated_resnet18(pretrained=True)
model, checkpoint = models_mlp.updated_resnet18(custom_pretrained=True)


# model = models.cnn_model()
# for name, param in model.named_parameters():
# if param.requires_grad:
#     print(name)
# if 'upd' in name:
#     param.requires_grad = False
# if 'layer' in name:
#     param.requires_grad= False
# if 'mlp' in name:
#     param.requires_grad = False
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
# # print(model.upd_conv1.weight.requires_grad=False)
# exit()

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
    plt.show()


#####################################


if False:
    x = torch.randn([1, 1, 256, 256])

    var = torch.autograd.Variable(x, requires_grad=True).to('cuda')
    model = model.to('cuda')
    model.eval()
    score, pred = model(var)

    pred = pred.view(-1, 4, 4).unsqueeze(1)
    print(pred.shape)
    output_point = torch.abs(pred[0, :, 1, 2]).sum()  # [0, :, 16, 32]

    grad = torch.autograd.grad(output_point, var)[0]
    grad[grad > 0] = 1

    print(grad.shape)

    grad_np = grad.detach().cpu().numpy()
    grad_np = grad_np[0, 0, 30, :]

    idx = np.where(grad_np != 0)[0]
    idx1 = np.where(grad_np[::-1] != 0)[0]

    print(idx[0])
    print(idx1[1])
    rf = 256 - idx[0] - idx1[1]
    print('receptive field size is:', rf)

    plt.plot(grad_np)
    plt.show()

    plot_tensor([grad])
    # print(pred.shape)

    print(grad)

    exit()

model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
# checkpoint = torch.load('/home/n-lab/PycharmProjects/contact-quality2/results/lab1_resnetv82_4xk/1_model.pth')
# model.module.load_state_dict(checkpoint['model'], strict=False)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# print(torchsummary.summary(model, (1, 256, 256), 32, device='cuda'))
# exit()
# learning parameters
lr = 2e-5
epochs = 40
# optimizer
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
# optimizer.load_state_dict(checkpoint['optimizer'])
# loss function

# exit()

criterion = nn.L1Loss()

# strings to save the loss plot, accuracy plot, and model with different ...
# ... names according to the training type
# if not using `--lr-scheduler` or `--early-stopping`, then use simple names
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'model'

# either initialize early stopping or learning rate scheduler
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer, patience=1)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    # model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping(patience=3)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    # model_name = 'es_model'


def compute_l2(x1, x2):
    bs = x1.shape[0]
    l = (x1 - x2).view(bs, -1).norm(2, 1).mean()
    return l


# training function
def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, (data, target, patch_class, patch_flag) in prog_bar:
        counter += 1
        # data, target = data[0].to(device), data[1].to(device)

        # w_batch = torch.tensor(w_train[target], dtype=float, device=device)
        # print(target, w_batch)
        # exit()
        data = data.float().to(device)
        target = target.float().to(device)

        patch_class = patch_class.float().to(device).squeeze(1)

        # patch_class = patch_class[patch_class == 1000]
        # print(patch_class)
        # exit()
        # patch_score = torch.tensor(patch_class, dtype=float, device=device)
        # patch_score = patch_class.view(-1, 4, 4)

        total += target.size(0)
        optimizer.zero_grad()
        outputs, score_map = model(data)

        # print(outputs.max(), outputs.min())

        loss1 = criterion(outputs.squeeze(1), target)
        # loss = compute_l2(outputs, target.unsqueeze(-1))
        # print(agg.shape, patch_score.shape)
        # exit()
        # loss1 = torch.abs(outputs - target)

        # print(loss.shape)

        # print(w_batch.shape)

        # loss1 = loss1 * w_batch

        # loss1 = loss1.mean()
        # print(classes.shape, patch_class.shape)
        # exit()
        patch_flag = torch.BoolTensor(patch_flag)
        patch_class = patch_class[patch_flag, :]
        score_map = score_map[patch_flag, :]
        if len(patch_class) == 0:
            loss = loss1
        else:
            loss2 = criterion(score_map, patch_class)
            loss = loss1 + 0.5*loss2

        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 1000:
            prog_bar.set_postfix(loss1=loss1.item())
        # print(train_running_loss/counter)

    train_loss = train_running_loss / counter
    # print(outputs.max(), outputs.min())
    return train_loss

# validation function
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset) / test_dataloader.batch_size))
    with torch.no_grad():
        for i, (data, target, patch_class, _) in prog_bar:
            counter += 1
            # data, target = data[0].to(device), data[1].to(device)

            # w_batch_val = torch.tensor(w_val[target], dtype=float, device=device)

            data = data.float().to(device)
            target = target.float().to(device)
            # patch_class = patch_class.float().to(device)

            # patch_score = torch.tensor(patch_score, dtype=float, device=device)
            # patch_score = patch_score.view(-1, 4, 4)

            total += target.size(0)
            outputs, score_map = model(data)

            # print(outputs.max(), outputs.min())
            loss = criterion(outputs.squeeze(1), target)
            # loss = compute_l2(outputs, target.unsqueeze(-1))
            # loss1 = torch.abs(outputs - target)
            # loss1 = loss1 * w_batch_val
            # loss1 = loss1.mean()

            # if patch_class != 1000:
            #     loss2 = criterion(score_map, patch_class)
            #     loss = loss1 + loss2
            # else:
            #     loss = loss1

            if i == 1:
                print(outputs, target)
                # print((classes.argmax(1, keepdim=True)), patch_class)

            val_running_loss += loss.item()


        val_loss = val_running_loss / counter
        # print(outputs.max(), outputs.min())
        return val_loss


# lists to store per-epoch loss and accuracy values
train_loss= []
val_loss = []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss = fit(
        model, train_dataloader, train_dataset, optimizer, criterion
    )
    val_epoch_loss = validate(
        model, val_dataloader, val_dataset, criterion
    )
    train_loss.append(train_epoch_loss)

    val_loss.append(val_epoch_loss)

    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')

end = time.time()
print(f"Training time: {(end - start) / 60:.3f} minutes")

exp_name = args['exp_name']
pathlib.Path(f'./results/{exp_name}').mkdir(parents=True, exist_ok=True)

# serialize the model to disk
print('Saving model...')
torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
           f"./results/{exp_name}/1_{model_name}.pth")

print('Saving loss plot...')


# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"./results/{exp_name}/{loss_plot_name}.png")
plt.show()

print('TRAINING COMPLETE')