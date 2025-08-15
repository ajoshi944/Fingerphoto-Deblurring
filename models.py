import torch.hub
import torch.nn as nn
import torchvision.models as models
from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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


def plot_tensor(t, scores, filename):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    fig, ax = plt.subplots(4, 5)

    for i, tensor in enumerate(t):
        ti_np = tensor2np_img(tensor)
        ax.ravel()[i].imshow(ti_np, cmap='gray')
        ax.ravel()[i].set_title(scores[i].cpu().detach().numpy())
    # fig.delaxes(ax[3][-3])
    fig.delaxes(ax[3][-2])
    fig.delaxes(ax[3][-1])
    plt.tight_layout()
    fig.suptitle(filename)
    # for i in range(len(t)):
    #     ti_np = tensor2np_img(t[i])
    #     plt.subplot(1, len(t), i + 1).set_title(scores[i])
    #     plt.imshow(ti_np, cmap='gray')
    plt.show()



#####################################
################################ RESNET18 #####################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, padding_mode='replicate') # padding = 1

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # out = self.tanh(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64 #64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False, padding_mode='replicate') # 64  padding=3
        self.bn1 = nn.BatchNorm2d(64) # 64
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) # 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1) # 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1) # 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 512

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(32 * block.expansion, num_classes)


        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1))#, nn.BatchNorm2d(128))

        self.fc1 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc3 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc4 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc5 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc6 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc7 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc8 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc9 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc10 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc11 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc12 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc13 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc14 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc15 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))
        self.fc16 = nn.Sequential(nn.Linear(1024, 1), nn.BatchNorm1d(1))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # self.vert1 = nn.Sequential(nn.Linear())

        self.fc_last = nn.Linear(16, 1)
        #self.bn2 = nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, gt_score=None, filename=None, return_map=False):
        orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        # print(x.max(), x.min())
        x = self.layer3(x)

        x = self.layer4(x)

        score_map = self.conv_1x1(x)
        # score_map = self.tanh(score_map)
        # print(score_map.max(), score_map.min())
        #score_map = self.bn2(score_map)
        #score_map = torch.sigmoid(score_map) * 100

        #final_score = score_map.mean([1, 2, 3])
        splits = F.unfold(score_map, kernel_size=32, stride=32)




        splits = splits.view(score_map.size(0), 32*32, 16)
        splits = splits.permute(0, 2, 1)



        # print(splits[:, 0].shape)
        # plot_tensor(splits[:, 0])
        # plot_tensor([score_map[0], splits[1][0], splits[1][1], splits[1][2], splits[1][3]])
        # exit()
        x1 = self.fc1(splits[:, 0])
        # dedicated batchnorm 1D
        x1 = self.sigmoid(x1)
        x2 = self.fc2(splits[:, 1])
        x2 = self.sigmoid(x2)
        x3 = self.fc3(splits[:, 2])
        x3 = self.sigmoid(x3)
        x4 = self.fc4(splits[:, 3])
        x4 = self.sigmoid(x4)
        x5 = self.fc5(splits[:, 4])
        x5 = self.sigmoid(x5)
        x6 = self.fc6(splits[:, 5])
        x6 = self.sigmoid(x6)
        x7 = self.fc7(splits[:, 6])
        x7 = self.sigmoid(x7)
        x8 = self.fc8(splits[:, 7])
        x8 = self.sigmoid(x8)
        x9 = self.fc9(splits[:, 8])
        x9 = self.sigmoid(x9)
        x10 = self.fc10(splits[:, 9])
        x10 = self.sigmoid(x10)
        x11 = self.fc11(splits[:, 10])
        x11 = self.sigmoid(x11)
        x12 = self.fc12(splits[:, 11])
        x12 = self.sigmoid(x12)
        x13 = self.fc13(splits[:, 12])
        x13 = self.sigmoid(x13)
        x14 = self.fc14(splits[:, 13])
        x14 = self.sigmoid(x14)
        x15 = self.fc15(splits[:, 14])
        x15 = self.sigmoid(x15)
        x16 = self.fc16(splits[:, 15])

        # print(x16.max(), x16.min())

        x16 = self.sigmoid(x16)
        # print(x16.max(), x16.min())
        # exit()
        # print(torch.squeeze(x16, dim=1).shape)
        # print(x16)
        # exit()
        agg = torch.stack([torch.squeeze(x1, dim=1), torch.squeeze(x2, dim=1), torch.squeeze(x3, dim=1),
                           torch.squeeze(x4, dim=1), torch.squeeze(x5, dim=1), torch.squeeze(x6, dim=1),
                           torch.squeeze(x7, dim=1), torch.squeeze(x8, dim=1), torch.squeeze(x9, dim=1),
                           torch.squeeze(x10, dim=1), torch.squeeze(x11, dim=1), torch.squeeze(x12, dim=1),
                           torch.squeeze(x13, dim=1), torch.squeeze(x14, dim=1), torch.squeeze(x15, dim=1),
                           torch.squeeze(x16, dim=1)], dim=1)


        final_score = self.fc_last(agg)




        # normalize
        final_score = torch.sigmoid(final_score)*100




        # plot_tensor([orig, score_map, splits[:, 0].view(-1, 32, 32), splits[:, 1].view(-1, 32, 32), splits[:, 2].view(-1, 32, 32),
        #              splits[:, 3].view(-1, 32, 32), splits[:, 4].view(-1, 32, 32), splits[:, 5].view(-1, 32, 32), splits[:, 6].view(-1, 32, 32),
        #              splits[:, 7].view(-1, 32, 32), splits[:, 8].view(-1, 32, 32), splits[:, 9].view(-1, 32, 32), splits[:, 10].view(-1, 32, 32),
        #              splits[:, 11].view(-1, 32, 32), splits[:, 12].view(-1, 32, 32), splits[:, 13].view(-1, 32, 32), splits[:, 14].view(-1, 32, 32),
        #              splits[:, 15].view(-1, 32, 32)], [gt_score, final_score,torch.squeeze(x1, dim=1), torch.squeeze(x2, dim=1),
        #                                               torch.squeeze(x3, dim=1), torch.squeeze(x4, dim=1), torch.squeeze(x5, dim=1), torch.squeeze(x6, dim=1),
        #                                               torch.squeeze(x7, dim=1), torch.squeeze(x8, dim=1), torch.squeeze(x9, dim=1), torch.squeeze(x10, dim=1), torch.squeeze(x11, dim=1), torch.squeeze(x12, dim=1),
        #                                                torch.squeeze(x13, dim=1), torch.squeeze(x14, dim=1), torch.squeeze(x15, dim=1), torch.squeeze(x16, dim=1)], filename)
        # print(agg)
        # print(final_score)
        # exit()
        # final_score = torch.mean(score_map.view(score_map.size(0), score_map.size(1), -1), dim=2)

        #
        # print(scores.shape) # bs x 1 x 64 x 64
        # exit()



        # 512x64x64




        # x = F.dropout(x, p=0.5)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # final_score = self.fc(x)

        if return_map:
            return final_score, score_map

        return final_score



def updated_resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.constant_(self.conv6.bias, 1)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')



    def forward(self, x, return_map=False):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.lrelu(x)
        x = self.conv6(x)
        score_map = self.relu(x)

        final_score = torch.mean(score_map.view(score_map.size(0), score_map.size(1), -1), dim=2)

        if return_map:
            return final_score, score_map

        return final_score


def cnn_model():
    model = CNNModel()
    return model