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
################################ MLP #####################################
class MLP(nn.Module):

    def __init__(self, input_size=128*128, output_size=1, layers=[64*64, 32*32, 256, 64]):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, layers[0])
        self.hidden2 = nn.Linear(layers[0], layers[1])
        self.hidden3 = nn.Linear(layers[1], layers[2])
        self.hidden4 = nn.Linear(layers[2], layers[3])
        self.output = nn.Linear(layers[3], output_size)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.bn2 = nn.BatchNorm1d(layers[1])
        self.bn3 = nn.BatchNorm1d(layers[2])
        self.bn4 = nn.BatchNorm1d(layers[3])


    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.relu(self.bn3(self.hidden3(x)))
        x = self.relu(self.bn4(self.hidden4(x)))
        self.out = self.relu(self.output(x))

        return self.out

class MLP_baseline2(nn.Module):

    def __init__(self, input_size=256, output_size=1, layers=[64, 16]):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, layers[0])
        # self.hidden2 = nn.Linear(layers[0], layers[1])
        self.output = nn.Linear(layers[0], output_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.bn2 = nn.BatchNorm1d(layers[1])


    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        # x = self.relu(self.bn2(self.hidden2(x)))
        self.out = self.relu(self.output(x))
        # self.out = self.output(x)
        # print(self.output.weight, self.output.bias)
        # exit()
        return self.out
###########################################################################
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
        self.inplanes = 64 # 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='replicate') # 64  padding=3
        self.bn1 = nn.BatchNorm2d(64) # 64
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) # 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1) # 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1) # 256
        #self.layer4 = self._make_layer(block, 128, layers[3], stride=1) # 512

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(32 * block.expansion, num_classes)


        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1))#, nn.BatchNorm2d(128))
        self.mlp = MLP_baseline2()
        # self.mlp1 = MLP()
        # self.mlp2 = MLP()
        # self.mlp3 = MLP()
        # self.mlp4 = MLP()
        # self.mlp5 = MLP()
        # self.mlp6 = MLP()
        # self.mlp7 = MLP()
        # self.mlp8 = MLP()
        # self.mlp9 = MLP()
        # self.mlp10 = MLP()
        # self.mlp11 = MLP()
        # self.mlp12 = MLP()
        # self.mlp13 = MLP()
        # self.mlp14 = MLP()
        # self.mlp15 = MLP()
        # self.mlp16 = MLP()


        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc_last = nn.Linear(16, 1)
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

        #x = self.layer4(x)
        """
        Uncomment for baseline 1.
        """
        # score_map = self.conv_1x1(x)
        # score_map = score_map.squeeze(dim=1).view(-1, 128*128)
        # final_score = self.mlp(score_map)
        # x = self.conv_1x1(x)
        x = x.mean(dim=(2,3))
        # print(x.shape)
        # exit()
        final_score = self.mlp(x)
        # final_score = nn.Sigmoid()(final_score)*100.
        # print(final_score.shape)
        # exit()
        # print(score_map.shape)
        # exit()

        # print(final_score.max(), final_score.min(), final_score.shape)
        # exit()
        # score_map = self.tanh(score_map)
        # print(score_map.max(), score_map.min())
        #score_map = self.bn2(score_map)
        #score_map = torch.sigmoid(score_map) * 100

        #final_score = score_map.mean([1, 2, 3])

        # splits = F.unfold(score_map, kernel_size=32, stride=32)
        # splits = splits.view(score_map.size(0), 32*32, 16)
        # splits = splits.permute(0, 2, 1)


        # print(splits[:, 0].shape)
        # plot_tensor(splits[:, 0])
        # plot_tensor([score_map[0], splits[1][0], splits[1][1], splits[1][2], splits[1][3]])
        # exit()
        # x1 = self.mlp1(splits[:, 0])
        # x2 = self.mlp2(splits[:, 1])
        # x3 = self.mlp3(splits[:, 2])
        # x4 = self.mlp4(splits[:, 3])
        # x5 = self.mlp5(splits[:, 4])
        # x6 = self.mlp6(splits[:, 5])
        # x7 = self.mlp7(splits[:, 6])
        # x8 = self.mlp8(splits[:, 7])
        # x9 = self.mlp9(splits[:, 8])
        # x10 = self.mlp10(splits[:, 9])
        # x11 = self.mlp11(splits[:, 10])
        # x12 = self.mlp12(splits[:, 11])
        # x13 = self.mlp13(splits[:, 12])
        # x14 = self.mlp14(splits[:, 13])
        # x15 = self.mlp15(splits[:, 14])
        # x16 = self.mlp16(splits[:, 15])

        # print(x16.shape)
        # print(x16.max(), x16.min())
        # exit()
        # print(torch.squeeze(x16, dim=1).shape)
        # print(x16)
        # exit()
        # agg = torch.stack([torch.squeeze(x1, dim=1), torch.squeeze(x2, dim=1), torch.squeeze(x3, dim=1),
        #                    torch.squeeze(x4, dim=1), torch.squeeze(x5, dim=1), torch.squeeze(x6, dim=1),
        #                    torch.squeeze(x7, dim=1), torch.squeeze(x8, dim=1), torch.squeeze(x9, dim=1),
        #                    torch.squeeze(x10, dim=1), torch.squeeze(x11, dim=1), torch.squeeze(x12, dim=1),
        #                    torch.squeeze(x13, dim=1), torch.squeeze(x14, dim=1), torch.squeeze(x15, dim=1),
        #                    torch.squeeze(x16, dim=1)], dim=1)


        # final_score = self.fc_last(agg)

        # final_weights = self.fc_last.weight
        # final_score = self.avgpool(agg)

        # normalize
        # final_score = torch.sigmoid(final_score)*100


        # plot_tensor([orig, agg.view(-1, 4, 4), splits[:, 0].view(-1, 32, 32), splits[:, 1].view(-1, 32, 32), splits[:, 2].view(-1, 32, 32),
        #              splits[:, 3].view(-1, 32, 32), splits[:, 4].view(-1, 32, 32), splits[:, 5].view(-1, 32, 32), splits[:, 6].view(-1, 32, 32),
        #              splits[:, 7].view(-1, 32, 32), splits[:, 8].view(-1, 32, 32), splits[:, 9].view(-1, 32, 32), splits[:, 10].view(-1, 32, 32),
        #              splits[:, 11].view(-1, 32, 32), splits[:, 12].view(-1, 32, 32), splits[:, 13].view(-1, 32, 32), splits[:, 14].view(-1, 32, 32),
        #              splits[:, 15].view(-1, 32, 32)], [gt_score, final_score, torch.squeeze(x1, dim=1), torch.squeeze(x2, dim=1),
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
            return final_score.squeeze(dim=1)

        return final_score.squeeze(dim=1)



def updated_resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model
