import torch.hub
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo
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


def split(im):
    im = im.squeeze(1)
    patches = []
    # ############################### Zoomed patches stride=64 window = 128 ##############################
    patch1 = im[:, 0:2, 0:2]
    patches.append(patch1)
    patch2 = im[:, 0:2, 2:4]
    patches.append(patch2)
    patch3 = im[:, 0:2, 4:6]
    patches.append(patch3)
    patch4 = im[:, 0:2, 6:8]
    patches.append(patch4)

    patch5 = im[:, 2:4, 0:2]
    patches.append(patch5)
    patch6 = im[:, 2:4, 2:4]
    patches.append(patch6)
    patch7 = im[:, 2:4, 4:6]
    patches.append(patch7)
    patch8 = im[:, 2:4, 6:8]
    patches.append(patch8)

    patch9 = im[:, 4:6, 0:2]
    patches.append(patch9)
    patch10 = im[:, 4:6, 2:4]
    patches.append(patch10)
    patch11 = im[:, 4:6, 4:6]
    patches.append(patch11)
    patch12 = im[:, 4:6, 6:8]
    patches.append(patch12)

    patch13 = im[:, 6:8, 0:2]
    patches.append(patch13)
    patch14 = im[:, 6:8, 2:4]
    patches.append(patch14)
    patch15 = im[:, 6:8, 4:6]
    patches.append(patch15)
    patch16 = im[:, 6:8, 6:8]
    patches.append(patch16)

    patches = torch.stack(patches)
    patches = patches.permute(1, 0, 2, 3)
    return patches


__all__ = ['ResNet', 'updated_resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

################################ MLP #####################################
class MLP(nn.Module):

    def __init__(self, input_size=4, output_size=1, layers=[5]):
        super().__init__()
        # self.hidden1 = nn.Linear(input_size, layers[0])
        # self.hidden2 = nn.Linear(layers[0], layers[1])
        self.output = nn.Linear(input_size, output_size)
        # self.relu = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm1d(layers[0])
        # self.bn2 = nn.BatchNorm1d(layers[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.relu(self.bn1(self.hidden1(x)))
        # x = self.relu(self.bn2(self.hidden2(x)))
        self.out = self.sigmoid(self.output(x))*100
        # self.out = self.output(x)
        return self.out

###########################################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.upd_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(64, 1)

        self.c1_1x1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.c2_1x1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.c3_1x1 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        # self.dense_mlp = nn.Sequential(nn.Conv2d(512, 1024, 1, 1, 0), nn.ReLU(),
        #                       nn.Conv2d(1024, 128, 1, 1, 0))
        self.split_mlps1 = SplitMLP(pretrained=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def forward(self, x, test=False):
        x = self.upd_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # x = F.normalize(x, 2, 1) # turn on this line for the last experiment

        x = self.c1_1x1(x) # bs x 256 x 8 x 8
        x = self.c2_1x1(x) # bs x 128 x 8 x 8
        x = self.c3_1x1(x) # bs x 1 x 8 x 8

        score_map, splits = self.split_mlps1(x) #score_map : (batch_size, 16)

        # score_map_avg = torch.mean(score_map, dim=1, keepdim=True) # bs,1


        # x = x.squeeze(1)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        x = torch.mean(x, dim=[1, 2])
        x = nn.Sigmoid()(x)*100

        # x = x.unsqueeze(1)
        # score = torch.mean(x, dim=[1, 2])
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # # x = self.relu(x)
        # # x = self.fc2(x)
        # score = nn.Sigmoid()(x)*100.
        if test:
            return x, score_map
        return x, score_map


class SplitMLP(nn.Module):
    def __init__(self, pretrained=False):
        super(SplitMLP, self).__init__()
        self.pretrained = pretrained

        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.mlp3 = MLP()
        self.mlp4 = MLP()
        self.mlp5 = MLP()
        self.mlp6 = MLP()
        self.mlp7 = MLP()
        self.mlp8 = MLP()
        self.mlp9 = MLP()
        self.mlp10 = MLP()
        self.mlp11 = MLP()
        self.mlp12 = MLP()
        self.mlp13 = MLP()
        self.mlp14 = MLP()
        self.mlp15 = MLP()
        self.mlp16 = MLP()

        if self.pretrained:
            mlp_list = [(self.mlp1, 'mlp1'), (self.mlp2, 'mlp2'), (self.mlp3, 'mlp3'), (self.mlp4, 'mlp4'),
                    (self.mlp5, 'mlp5'), (self.mlp6, 'mlp6'), (self.mlp7, 'mlp7'), (self.mlp8, 'mlp8'),
                    (self.mlp9, 'mlp9'), (self.mlp10, 'mlp10'), (self.mlp11, 'mlp11'), (self.mlp12, 'mlp12'),
                    (self.mlp13, 'mlp13'), (self.mlp14, 'mlp14'), (self.mlp15, 'mlp15'), (self.mlp16, 'mlp16')]

            for _, (mlp, mlp_name) in enumerate(mlp_list):
                self.load_weights(mlp, mlp_name)



    def load_weights(self, mlp, mlp_name):
        checkpoint = torch.load(f'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vin_c10_mean_manual_labeling_v3/1_model.pth')
        mlp.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, score_map):
        # splits = F.unfold(score_map, kernel_size=32, stride=32)
        # local scores code
        # splits = F.unfold(score_map, kernel_size=2, stride=2, padding=0)
        splits = split(score_map)
        splits = splits.view(score_map.size(0), 16, 2*2)
        # splits = splits.permute(0, 2, 1)

        # print(splits[:, 0].shape)
        # plot_tensor(splits[:, 0])
        # plot_tensor([score_map[0], splits[1][0], splits[1][1], splits[1][2], splits[1][3]])
        # exit()
        # with torch.no_grad():
        x1 = self.mlp1(splits[:, 0])
        x2 = self.mlp2(splits[:, 1])
        x3 = self.mlp3(splits[:, 2])
        x4 = self.mlp4(splits[:, 3])
        x5 = self.mlp5(splits[:, 4])
        x6 = self.mlp6(splits[:, 5])
        x7 = self.mlp7(splits[:, 6])
        x8 = self.mlp8(splits[:, 7])
        x9 = self.mlp9(splits[:, 8])
        x10 = self.mlp10(splits[:, 9])
        x11 = self.mlp11(splits[:, 10])
        x12 = self.mlp12(splits[:, 11])
        x13 = self.mlp13(splits[:, 12])
        x14 = self.mlp14(splits[:, 13])
        x15 = self.mlp15(splits[:, 14])
        x16 = self.mlp16(splits[:, 15])

        # print(x16.shape)
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

        # agg = agg.permute(0, 2, 1)
        return agg, splits

def updated_resnet18(pretrained=False, custom_pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    elif custom_pretrained:
        checkpoint = torch.load(f"/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vin_mean_c10_v17/1_model.pth") #/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_mod_v8.2/1_model.pth
        model.load_state_dict(checkpoint['model'], strict=False)
        return model, checkpoint
    return model



def resnet34(pretrained=False, custom_pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    elif custom_pretrained:
        checkpoint = torch.load(f"/home/n-lab/PycharmProjects/contact-quality2/results/lab1_resnet34_baseline/1_model.pth")
        model.load_state_dict(checkpoint['model'], strict=False)
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

