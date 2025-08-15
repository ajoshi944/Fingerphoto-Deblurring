import torch.hub
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import matplotlib
from torch.nn.modules.module import T
from att_models.cbam import CBAM

#####################################
def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


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
            ti_np = tensor2np_img(tensor, normalize=False)
            ax.ravel()[i].imshow(ti_np, cmap=viridis_respampled, vmin=0.0, vmax=100.)
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

def plot_tensor_basic(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = tensor2np_img(t[i])
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()

def split(im):
    im = im.squeeze(1)
    patches = []
    # ############################### Zoomed patches stride=64 window = 128 ##############################
    # patch1 = im[:, 0:3, 0:3]
    # patches.append(patch1)
    # patch2 = im[:, 0:3, 2:5]
    # patches.append(patch2)
    # patch3 = im[:, 0:3, 3:6]
    # patches.append(patch3)
    # patch4 = im[:, 0:3, 5:8]
    # patches.append(patch4)
    #
    # patch5 = im[:, 2:5, 0:3]
    # patches.append(patch5)
    # patch6 = im[:, 2:5, 2:5]
    # patches.append(patch6)
    # patch7 = im[:, 2:5, 3:6]
    # patches.append(patch7)
    # patch8 = im[:, 2:5, 5:8]
    # patches.append(patch8)
    #
    # patch9 = im[:, 3:6, 0:3]
    # patches.append(patch9)
    # patch10 = im[:, 3:6, 2:5]
    # patches.append(patch10)
    # patch11 = im[:, 3:6, 3:6]
    # patches.append(patch11)
    # patch12 = im[:, 3:6, 5:8]
    # patches.append(patch12)
    #
    # patch13 = im[:, 5:8, 0:3]
    # patches.append(patch13)
    # patch14 = im[:, 5:8, 2:5]
    # patches.append(patch14)
    # patch15 = im[:, 5:8, 3:6]
    # patches.append(patch15)
    # patch16 = im[:, 5:8, 5:8]
    # patches.append(patch16)

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

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, ref_model=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # self.fc = nn.Sequential(nn.Linear(2048, 512),
        #                         nn.ReLU())
        # self.fc1 = nn.Linear(64, 1)
        # self.fc_big1 = nn.Linear(8192, 2048)
        # self.fc2 = nn.Linear(2048, 1)

        # self.c1_1x1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256),
        #                             nn.ReLU(inplace=True))
        # self.c2_1x1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128),
        #                             nn.ReLU(inplace=True))
        # self.c3_1x1 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        # self.CBAM1 = CBAM(64*block.expansion, spatial=False)
        # self.CBAM2 = CBAM(128*block.expansion, spatial=False)
        # self.CBAM3 = CBAM(256*block.expansion, spatial=False)

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

    def forward(self, x, mask=None, pool=False):
        bs = x.shape[0]
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.CBAM1(x)
        x = self.layer2(x)
        # x = self.CBAM2(x)
        x = self.layer3(x)
        # x = self.CBAM3(x)
        x = self.layer4(x)
        # x = F.normalize(x, 2, 1) # turn on this line for the last experiment
        if pool:
            x = self.avgpool(x)
        # x = x.view(bs, -1)
        # x = self.fc(x)

        # plot_tensor_basic([input, x, attn_weights])

        return x#.unsqueeze(2).unsqueeze(3)

class SelfAttN(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256):
        super(SelfAttN, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, candidate, reference, seed=None):
        F = self.f(candidate)
        G = self.g(reference)
        H = self.h(reference)
        b, dim, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(candidate.device)[:self.max_sample]
            G = G[:, :, index]
            ref_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            ref_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1) / (dim ** 0.5)
        # print(F.shape, G.shape)
        attention = torch.bmm(F, G)
        # print(attention.shape)
        # b, n_c, n_s
        weights = self.sm(attention)
        # print(weights.shape, ref_flat.shape)
        # b, n_c, c
        weighted = torch.bmm(weights, ref_flat)
        # print(weighted.shape)
        # exit()
        weighted = weighted.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return weighted

class Fuse(nn.Module):
    def __init__(self, args):
        super(Fuse, self).__init__()
        self.args = args
        self.att = SelfAttN(512)
        self.norm = nn.LayerNorm(512)
        # self.conv1 = nn.Conv2d(512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mlp = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256),
        #                             nn.GELU(),
        #                          nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128),
        #                             nn.GELU(),
        #                          nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0))
        # self.mlp = nn.Sequential(nn.Linear(512, 128), )
        # self.conv = nn.Conv2d(512, 1, kernel_size=1)
        self.fc = nn.Linear(512, 1)
        # self.fc2 = nn.Linear(512, 1)
        self.criterion = nn.MSELoss().cuda()
        self.criterion2 = nn.CosineEmbeddingLoss().cuda()
        # self.criterion_BCE = nn.BCEWithLogitsLoss()

    def forward(self, candidate_feat, reference_feat, matching_score, label):
        attended = self.att(candidate_feat, reference_feat)
        # attended += candidate_feat
        # x = self.mlp(attended) # bs x 256 x 8 x 8
        # attended = self.conv(attended)
        attended = self.avgpool(attended)
        x = attended.view(attended.shape[0], -1)
        normed_x = self.norm(x)

        score = self.fc(normed_x)
        # pair = self.fc2(x)
        # score = torch.clip(score, 0, 1)
        # pair = torch.clip(pair, 0, 1)
        loss1 = self.criterion(score.squeeze(1), matching_score) * 10.0
        label = (label - 0.5) / 0.5
        loss4 = self.criterion2(candidate_feat.view(label.shape[0], -1), reference_feat.view(label.shape[0], -1), label) * 2.0
        # loss4 = arcface_loss(x, label) * 0.5
        # loss2 = self.criterion_BCE(pair.squeeze(1), label) * 0.

        # loss = loss1 + loss4 #+ loss2
        return loss1, loss4#loss,

class Quality(nn.Module):
    def __init__(self):
        super(Quality, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(512, 1024),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 2048),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(2048),
                                 nn.Linear(2048, 4096))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv2d(512, 1, kernel_size=1)
        # self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()
        self.criterion = nn.MSELoss().cuda()

    def forward(self, x, target=None, cand_qm=None):
        x = self.avgpool(x)
        # x = self.conv(x)
        # x = self.relu(x)
        x = x.view(x.shape[0], -1)

        # qm = None
        qm = self.mlp(x)
        # qm = torch.clip(qm, 0., 1.)
        qm_res = qm.view(x.shape[0], 1, 64, 64)

        # qm = self.relu(qm)
        score = self.fc(x)
        # print(score)
        score = self.sig(score) * 100.
        if target is not None or cand_qm is not None:
            loss = self.criterion(score.squeeze(1), target) * 0.1
            qm_loss = (self.criterion(qm_res, cand_qm) +
                       self.criterion(torch.mean(qm_res, dim=(1, 2, 3)), torch.mean(cand_qm, dim=(1, 2, 3))) +
                       self.criterion(torch.std(qm_res, dim=(1, 2, 3)), torch.std(cand_qm, dim=(1, 2, 3)))) * 10.
            return loss, qm_loss
        return score, qm_res

class NfiqQuality(nn.Module):
    def __init__(self):
        super(NfiqQuality, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(512, 128),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(128, 32),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(32),
                                 nn.Linear(32, 1))

        # self.fc = nn.Linear(4096, 1)
        self.sig = nn.Sigmoid()
        self.criterion = nn.MSELoss().cuda()

    def forward(self, x, target=None):
        x = x.view(x.shape[0], -1)
        # qm = None
        score = self.mlp(x)
        # score = torch.clip(score, 0., 1.)
        # score = self.fc(qm)
        # score = self.sig(score)# * 100.
        if target is not None:
            loss = self.criterion(score.squeeze(1), target) * 0.1
            return loss
        return score


def updated_resnet18(pretrained=False, custom_pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], zero_init_residual=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    elif custom_pretrained:
        # checkpoint = torch.load(f"/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vin_mean_c10_v17/1_model.pth") #/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_mod_v8.2/1_model.pth
        checkpoint = torch.load(
            f"/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/lab1_resnetv82_4xk/1_model.pth")
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

