import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from . import model

class ForwardHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = {}
        self.input = {}

    def hook_fn(self, module, input, output):
        device = input[0].device
        self.output[device] = output
        self.input[device] = input[0]

    def close(self):
        self.hook.remove()

def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = t[i].cpu().detach().numpy().squeeze()
        ti_np = norm_minmax(ti_np)
        if len(ti_np.shape) > 2:
            ti_np = ti_np.transpose(1, 2, 0)
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf

    BLUR TYPE ESTIMATION IS ADDED TO THIS CODE.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_GAN2', 'G_GAN3','G_L1', 'G_L1_2', 'G_L1_3', 'D_real', 'D_fake', 'att_128', 'att_64', 'verifier', 'verif_features', 'blurring_type']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B']#['real_A', 'fake_B', 'real_B', 'int_fm_64_3_256','int_fm_128_3_256', 'vis_att_64_3_256','vis_att_128_3_256', 'mask_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'D2', 'D3']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #gen_checkpoint = torch.load('/home/n-lab/Amol/deblurring/contactless/checkpoints/clmixpv15.4/200_net_G.pth')
        #self.netG.module.load_state_dict(gen_checkpoint['model'])
        #print('Generetor Loaded!')
        self.netG.eval()
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #self.netD.eval()
            #desc_checkpoint = torch.load('/home/n-lab/Amol/deblurring/contactless/checkpoints/clmixpv15.3/200_net_D.pth')
            #self.netD.module.load_state_dict(desc_checkpoint['model'])

            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #desc2_checkpoint = torch.load('/home/n-lab/Amol/deblurring/contactless/checkpoints/clmixpv15.3/200_net_D2.pth')
            #self.netD2.module.load_state_dict(desc2_checkpoint['model'])

            self.netD3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #desc3_checkpoint = torch.load('/home/n-lab/Amol/deblurring/contactless/checkpoints/clmixpv15.3/200_net_D3.pth')
            #self.netD3.module.load_state_dict(desc3_checkpoint['model'])

            #print('Discriminators Loaded!')
        #conv_checkpoint = torch.load('/home/n-lab/Amol/deblurring/contactless/checkpoints/cl3motionv7_drv_stack_intsatt/200_net_G.pth')
        # self.conv_1x1_128 = torch.nn.Sequential(torch.nn.Conv2d(128, 3, 1),
        #                                         nn.Tanh())  # 1x1 convolution to change the number of channels
        # if len(self.gpu_ids) > 0:
        #     assert (torch.cuda.is_available())
        #     self.conv_1x1_128.to(self.gpu_ids[0])
        #     self.conv_1x1_128 = torch.nn.DataParallel(self.conv_1x1_128, self.gpu_ids)  # multi-GPUs
        #
        # self.conv_1x1_64 = torch.nn.Sequential(torch.nn.Conv2d(256, 3, 1),
        #                                        nn.Tanh())  # 1x1 convolution to change the number of channels
        # if len(self.gpu_ids) > 0:
        #     assert (torch.cuda.is_available())
        #     self.conv_1x1_64.to(self.gpu_ids[0])
        #     self.conv_1x1_64 = torch.nn.DataParallel(self.conv_1x1_64, self.gpu_ids) # multi-GPUs

        if self.isTrain:
            #self.conv_1x1_64.module.load_state_dict(conv_checkpoint['conv64'])
            #self.conv_1x1_128.module.load_state_dict(conv_checkpoint['conv128'])
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters())+list(self.netD2.parameters())+list(self.netD3.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G.load_state_dict(gen_checkpoint['optimizer'])
            #self.optimizer_D.load_state_dict(desc_checkpoint['optimizer'])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # Load Verifier
        self.net_v = model.Verifier().to(self.device)
        checkpoint = torch.load('/home/n-lab/Amol/model_resnet18_100.pt')
        self.net_v.load_state_dict(checkpoint['net_photo'])
        self.net_v.eval()
        print('Verifier Ready')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.mask = input['mask'].to(self.device)

        if self.isTrain:
            self.blurring_type = input['blurring_type'].to(self.device)

        # bs = len(self.real_A)
        # for i in range(bs):
        #     print(self.blurring_type[i].item())
        #     plot_tensor([self.real_A[i]])
        # exit()
        #self.bt = torch.zeros([self.real_A.shape[0], 1, self.real_A.shape[2], self.real_A.shape[3]], device=self.device)

        #self.blurring_type_mask = torch.zeros([self.real_A.shape[0], 1, self.real_A.shape[2], self.real_A.shape[3]], device=self.device)
        #self.real_A = torch.cat([self.real_A, self.blurring_type_mask], dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.gen_out = self.netG(self.real_A)  # G(A)

        self.fake_B = self.gen_out[0]
        # Get the intermediate features to compute reconstruction loss
        # self.int_fm_128_3 = self.gen_out[1]
        # #self.int_fm_128_3 = self.conv_1x1_128(self.int_fm_128)
        # self.int_fm_128_3_256 = F.interpolate(self.int_fm_128_3, 256, mode='bilinear', align_corners=False)
        # self.real_B_128 = F.interpolate(self.real_B, 128, mode='bilinear', align_corners=False) # mode = 'nearest'
        # self.real_B_128.to('cuda:1')
        # # for 64x64
        # self.int_fm_64_3 = self.gen_out[3]
        # #self.int_fm_64_3 = self.conv_1x1_64(self.int_fm_64)
        # self.int_fm_64_3_256 = F.interpolate(self.int_fm_64_3, 256, mode='bilinear', align_corners=False)
        # self.real_B_64 = F.interpolate(self.real_B, 64, mode='bilinear', align_corners=False) # mode = 'nearest'
        # self.real_B_64.to('cuda:1')
        # # Visualize Attentions
        # self.att_64 = self.gen_out[4].repeat([1, 3, 1, 1]) # output of 1x1 in the spatial att
        # self.att_128 = self.gen_out[2].repeat([1, 3, 1, 1]) # output of 1x1 in the spatial att
        # #self.vis_int_64 = self.gen_out[2] # output of 1x1 in the spatial att
        # self.att_64_3_256 =  F.interpolate(self.att_64, 256, mode='bilinear', align_corners=False)
        # self.vis_att_64_3_256 = self.att_64_3_256 * 2 - 1
        # self.att_128_3_256 =  F.interpolate(self.att_128, 256, mode='bilinear', align_corners=False)
        # self.vis_att_128_3_256 = self.att_128_3_256 * 2 - 1
        #
        # self.mask_vis = self.mask * 2 - 1
        # #print(self.mask.max(), self.mask.min())\
        # #exit()
        #
        # self.predicted_type = self.gen_out[-1]
        # self.predicted_types = self.gen_out[-1].argmax(dim=1)
        # self.blurring_types = self.predicted_types.data

    def ridge_visuals(self):
        self.real_ridge = self.net_re(
            self.real_B).detach()  # detach the real_ridge from the computation graph because it is our target
        self.generated_ridge = self.net_re(self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #real_A = self.real_A[:,0:3,:,:]
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator

        self.real_A_128 = F.interpolate(self.real_A, 128, mode='bilinear', align_corners=False)
        fake_AB2 = torch.cat((self.real_A_128, self.int_fm_128_3), 1)

        self.real_A_64 = F.interpolate(self.real_A, 64, mode='bilinear', align_corners=False)
        fake_AB3 = torch.cat((self.real_A_64, self.int_fm_64_3), 1)

        pred_fake3 = self.netD3(fake_AB3.detach())
        pred_fake2 = self.netD2(fake_AB2.detach())
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake3 = self.criterionGAN(pred_fake3, False)
        self.loss_D_fake2 = self.criterionGAN(pred_fake2, False)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB3 = torch.cat((self.real_A_64, self.real_B_64), 1)
        real_AB2 = torch.cat((self.real_A_128, self.real_B_128), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real3 = self.netD3(real_AB3)
        pred_real2 = self.netD2(real_AB2)
        pred_real = self.netD(real_AB)
        self.loss_D_real3 = self.criterionGAN(pred_real3, True)
        self.loss_D_real2 = self.criterionGAN(pred_real2, True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D3 = (self.loss_D_fake3 + self.loss_D_real3) * 0.5
        self.loss_D2 = (self.loss_D_fake2 + self.loss_D_real2) * 0.5
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        ((self.loss_D + self.loss_D2 + self.loss_D3)/(0.5 + 0.5 + 0.5)).backward()

    def compute_l2(self, x1, x2):
        bs = x1.shape[0]
        l = (x1-x2).view(bs, -1).norm(2, 1).mean()
        return l

    def backward_G(self):
        # Verification loss
        with torch.no_grad():
            self.real_embedding, self.real_features = self.net_v(self.real_B)
        self.deblurred_embedding, self.deblurred_features = self.net_v(self.fake_B)

        self.loss_verif_features = 0
        lambda_ridge_features = [1.0, 1.0, 1.0]

        for i in range(3):
            f_real = self.real_features[i]
            f_deblurred = self.deblurred_features[i]

            self.loss_verif_features += self.compute_l2(f_real, f_deblurred) * lambda_ridge_features[i]
        lambda_verifier = 0.01
        loss_verif = (self.deblurred_embedding - self.real_embedding) ** 2
        self.loss_verifier = loss_verif.sum(1).mean()
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #real_A = self.real_A[:, 0:3, :, :]
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        fake_AB2 = torch.cat((self.real_A_128, self.int_fm_128_3), 1)
        pred_fake2 = self.netD2(fake_AB2)

        fake_AB3 = torch.cat((self.real_A_64, self.int_fm_64_3), 1)
        pred_fake3 = self.netD3(fake_AB3)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN2 = self.criterionGAN(pred_fake2, True)
        self.loss_G_GAN3 = self.criterionGAN(pred_fake3, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_L1_2 = self.criterionL1(self.int_fm_128_3, self.real_B_128) * self.opt.lambda_L1
        self.loss_G_L1_3 = self.criterionL1(self.int_fm_64_3, self.real_B_64) * self.opt.lambda_L1
        # Compute reconstruction loss for attention with mask
        self.loss_att_128 = self.criterionL1(self.att_128_3_256, self.mask) * self.opt.lambda_L1
        self.loss_att_64 = self.criterionL1(self.att_64_3_256, self.mask) * self.opt.lambda_L1
        # Blurring type loss
        #print((torch.mean(self.gen_out[-1], dim=[1,2,3])).shape)
        #print(self.gen_out[-1].shape, self.blurring_type.shape)
        #exit()
        # self.loss_blurring_type = nn.BCEWithLogitsLoss()(self.gen_out[-1], self.blurring_type.float())
        self.loss_blurring_type = nn.CrossEntropyLoss()(self.predicted_type, self.blurring_type)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN * 0.6 + self.loss_G_GAN2 * 0.4 + self.loss_G_GAN3 * 0.1 + self.loss_G_L1 * 0.6 + self.loss_G_L1_2 * 0.4 + self.loss_G_L1_3 * 0.1 + self.loss_att_128 * 1.2 + self.loss_att_64 * 0.3 + self.loss_verifier * lambda_verifier + self.loss_verif_features + self.loss_blurring_type*1.2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netD2, True) # enable backprop for D2
        self.set_requires_grad(self.netD3, True) # enable backprop for D3
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2, False)
        self.set_requires_grad(self.netD3, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
