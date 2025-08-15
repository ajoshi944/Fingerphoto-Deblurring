import json
import math
import pathlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib
import matplotlib.pyplot as plt
import time
# import models_mlp_binary_posencoding as models_mlp
# import models_mlp_backup as models_mlp
import resnet18_dual_w_SA as models_mlp
import argparse
import sys
from dataset_dual import get_datasets
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
from torch.distributed import init_process_group
import torchsummary
import numpy as np
# matplotlib.style.use('ggplot')
from pytorch_metric_learning import losses
from lr_scheduler import PolynomialLRWarmup
# from partial_fc_v2 import PartialFC_V2
# from losses import CombinedMarginLoss
"""
Command for training:
python train.py --exp_name vgg_5ksize_v5

train_feat_regressor
"""
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--exp_name', required=True)
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=102, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')

def main():
    args = parser.parse_args()
    args.ngpus_per_node = 1#torch.cuda.device_count()
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    # torch.set_num_threads(12 // args.world_size)
    # print(torch.multiprocessing.cpu_count())
    # exit()
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)#
    # main_worker()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Computation device: {device}\n")

def main_worker(gpu, args):
    args.rank += gpu
    print(gpu, args.rank)
    # print(os.getloadavg()[0])
    init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size,
                                         rank=args.rank)
    if args.rank == 0:
        pathlib.Path(f'./results/{args.exp_name}').mkdir(parents=True, exist_ok=True)
        stats_file = open(f'./results/{args.exp_name}/stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    # instantiate the model
    """
    Change the method to get different models.
    exp_resnet18(): Complete resnet18 architecture with kernel size = 5 and # of kernels 16->32->64->128
    get_vgg(): VGG11_bn architecture with kernel size = 5 and # of kernels 16->32->64->128
    """
    model = models_mlp.updated_resnet18()#models_mlp.updated_resnet18()#(models_mlp.updated_resnet18(args, pretrained=True))
    model = model.cuda(gpu)
    kwargs = {'ref_model': True}
    ref_model = models_mlp.updated_resnet18()#models_mlp.updated_resnet18(**kwargs)
    ref_model = ref_model.cuda(gpu)
    fusing_model = models_mlp.Fuse(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ref_model = nn.SyncBatchNorm.convert_sync_batchnorm(ref_model)
    quality_model = models_mlp.Quality().cuda(gpu)
    ckpt = torch.load('/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/quality_new_v8_1_ish/latest_model.pth')
    model.load_state_dict(ckpt['model'])
    ref_model.load_state_dict(ckpt['ref_model'])
    fusing_model.load_state_dict(ckpt['fusing_model'])
    quality_model.load_state_dict(ckpt['quality_model'])
    # nfiq_model = models_mlp.NfiqQuality().cuda(gpu)

    # optimizer
    # param_weights = []
    # param_biases = []
    # for param in model.parameters():
    #     if param.ndim == 1:
    #         param_biases.append(param)
    #     else:
    #         param_weights.append(param)
    # for param in fusing_model.parameters():
    #     if param.ndim == 1:
    #         param_biases.append(param)
    #     else:
    #         param_weights.append(param)
    # parameters = [{'params': param_weights}, {'params': param_biases}]
    # arcface_loss = losses.ArcFaceLoss(num_classes=2, embedding_size=512, margin=28.6, scale=64).to(torch.device('cuda'))
    # margin_loss = CombinedMarginLoss(64,
    #                                  1.0,
    #                                  0.5,
    #                                  0.0,
    #                                  0)
    # module_partial_fc = PartialFC_V2(margin_loss, 512, 2, 1, False)
    parameters = (list(model.parameters()) + list(ref_model.parameters()) +list(fusing_model.parameters()) + list(quality_model.parameters()))#+ list(nfiq_model.parameters()))  +  list(quality_model.parameters())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], bucket_cap_mb=16)
    ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, device_ids=[gpu], bucket_cap_mb=16)
    fusing_model = torch.nn.parallel.DistributedDataParallel(fusing_model, device_ids=[gpu], bucket_cap_mb=16)
    quality_model = torch.nn.parallel.DistributedDataParallel(quality_model, device_ids=[gpu], bucket_cap_mb=16)
    # nfiq_model = torch.nn.parallel.DistributedDataParallel(nfiq_model, device_ids=[gpu])
    optimizer = optim.Adam(parameters, lr=0.0002, weight_decay=5e-4)#optim.SGD(parameters, lr=0.02, momentum=0.9, weight_decay=5e-4)
    # optimizer.load_state_dict(ckpt['optimizer'])


    start_epoch = 1

    train_dataset, _ = get_datasets(clf=False, train=True, weights=False, batch_size=args.batch_size)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size,
                                                   num_workers=args.workers, pin_memory=True,
                                                   sampler=sampler, drop_last=True)

    warmup_step = len(train_dataset) // args.batch_size * 0
    total_step = len(train_dataset) // args.batch_size * args.epochs

    lr_scheduler = PolynomialLRWarmup(optimizer=optimizer,
                                      warmup_iters=warmup_step,
                                      total_iters=total_step)
    # lists to store per-epoch loss and accuracy values
    train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = [], [], [], [], []
    model_name = 'model'
    start = time.time()
    scaler = torch.cuda.amp.GradScaler(growth_interval=100)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        # adjust_learning_rate(args, optimizer, train_dataloader, epoch=epoch)
        for step, data_dict in enumerate(train_dataloader):#, start=epoch*len(train_dataloader)):
            model.eval()
            ref_model.train()
            fusing_model.train()
            quality_model.train()
            # nfiq_model.train()
            train_running_loss = 0.0
            train_running_correct = 0
            counter = 0
            total = 0
            transfer = lambda x: x.float().cuda(gpu, non_blocking=True)
            counter += 1

            candidate = transfer(data_dict['candidate'])
            reference = transfer(data_dict['reference'])
            # target = transfer(data_dict['target'])
            matching_score = transfer(data_dict['matching_score'])
            cand_innq = transfer(data_dict['cand_innq'])
            ref_innq = transfer(data_dict['ref_innq'])
            # cand_nfiq = transfer(data_dict['cand_nfiq'])
            # ref_nfiq = transfer(data_dict['ref_nfiq'])
            label = transfer(data_dict['label'])#data_dict['label'].type(torch.float).cuda(gpu, non_blocking=True)
            candidate_qm = transfer(data_dict['candidate_qm'])
            reference_qm = transfer(data_dict['reference_qm'])
            # cand_sh = transfer(data_dict['candidate_sh'])
            # ref_sh = transfer(data_dict['reference_sh'])
            cand_ivniq = transfer(data_dict['candidate_ivniq'])
            # ref_ivniq = transfer(data_dict['ref_ivniq'])
            sh_ratio = transfer(data_dict['sh_ratio'])
            target = torch.round(((0.5 * (cand_ivniq + cand_innq)) * sh_ratio))
            # target = (cand_ivniq + cand_sh) * 0.5
            # target = torch.min(cand_ivniq, cand_innq)
            # target_ref = torch.min(ref_ivniq, ref_innq)
            # total += target.size(0)
            # adjust_learning_rate(args, optimizer, train_dataloader, step, epoch, warmup=True)
            # if target[2] > 90.:
            #     print(data_dict['candidate_name'][2], target[2])
            #     plot_tensor([candidate[2]])
            # continue
            with torch.cuda.amp.autocast():
            # with torch.no_grad():
                candidate_feats = model(candidate, True)
                # with torch.no_grad():
                reference_feats = ref_model(reference, True)
                # print(candidate_feats.shape, reference_feats.shape)
                # exit()
                loss1, loss4 = fusing_model(candidate_feats, reference_feats, matching_score, label)
                # print(comb_feats.dtype, label.dtype)
                # exit()
            # loss4: torch.Tensor = module_partial_fc(comb_feats, label)*0.01

                # cand_ref_feats = model(reference, True)
                loss2, loss3 = quality_model(candidate_feats,
                                             target,
                                             candidate_qm)
                # loss2, loss3 = quality_model(torch.cat((candidate_feats, reference_feats), dim=0),
                #                              torch.cat((target, target_ref), dim=0),
                #                              torch.cat((candidate_qm, reference_qm), dim=0))
                # loss4, loss5 = quality_model(reference_feats, ref_innq, reference_qm)
                # loss5 = nfiq_model(torch.cat((candidate_feats, reference_feats), dim=0),
                #                              torch.cat((cand_nfiq, ref_nfiq), dim=0))
                # loss6 = nfiq_model(reference_feats, ref_nfiq)
            loss = loss1 + loss4 + loss2 + loss3#   #+ loss5 loss2 ++ loss3
            scaler.scale(loss).backward()

            if step % 1 == 0:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                # torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 5)
                # torch.nn.utils.clip_grad_norm_(fusing_model.parameters(), 5)
                # torch.nn.utils.clip_grad_norm_(quality_model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            lr_scheduler.step()



            train_running_loss += loss.item()

                # preds = torch.round(torch.sigmoid(outputs))
                # train_running_correct += (preds == target).sum().item()


            if step % 100 == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step, lr_weights=lr_scheduler.get_last_lr()[0],#optimizer.param_groups[0]['lr'],
                                 loss1=loss1.item(), loss4=loss4.item(), loss2=loss2.item(), loss3=loss3.item(),   # lr_biases=optimizer.param_groups[1]['lr'],loss5=loss5.item()
                                 time=int(time.time() - start))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            train_loss1.append(loss1.item())
            train_loss2.append(loss2.item())
            train_loss3.append(loss3.item())
            train_loss4.append(loss4.item())
            # serialize the model to disk
            print('Saving model...')
            torch.save({'model': model.module.state_dict(),
                        'ref_model': ref_model.module.state_dict(),
                        'fusing_model': fusing_model.module.state_dict(),
                        'quality_model': quality_model.module.state_dict(),
                        # 'nfiq_model': nfiq_model.module.state_dict(),
                        'optimizer': optimizer.state_dict()}, f"./results/{args.exp_name}/{epoch}_{model_name}.pth")
                # print(train_running_loss/counter)
    if args.rank == 0:
        print('Saving final model...')
        torch.save({'model': model.module.state_dict(),
                    'ref_model': ref_model.module.state_dict(),
                    'fusing_model': fusing_model.module.state_dict(),
                    'quality_model': quality_model.module.state_dict(),
                    # 'nfiq_model': nfiq_model.module.state_dict(),
                    'optimizer': optimizer.state_dict()}, f"./results/{args.exp_name}/latest_{model_name}.pth")

    end = time.time()
    print(f"Training time: {(end - start) / 60:.3f} minutes")
    if args.rank == 0:
        loss_plot_name = 'loss'
        print('Saving loss plots...')
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss1, color='orange', label='train loss 1')
        plt.plot(train_loss2, color='green', label='train loss 2')
        plt.plot(train_loss3, color='brown', label='train loss 3')
        plt.plot(train_loss4, color='red', label='train loss 4')
        # plt.plot(train_loss5, color='olive', label='train loss 5')
        # plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"./results/{args.exp_name}/{loss_plot_name}.png")
        plt.show()

    print('TRAINING COMPLETE')



def adjust_learning_rate(args, optimizer, loader, step=None, epoch=None, warmup=False):
    total_batches = len(loader)
    # max_steps = args.epochs * total_batches # 20 * 280
    warmup_epochs = 3
    warmup_steps = warmup_epochs * total_batches # 3 * 280
    # base_lr = args.batch_size / 17#256
    if warmup and epoch < warmup_epochs:
        eta_min = 0.005 * (0.1 ** 3)
        warmup_to = eta_min + (0.005 - eta_min) * (
                    1 + math.cos(math.pi * 3 / args.epochs)) / 2
        p = (step + (epoch - 1) * total_batches) / warmup_steps
        lr = 0.001 + p * (warmup_to - 0.001)
        # lr = base_lr * step / warmup_steps
    else:
        lr = 0.005
        eta_min = lr * (0.1 ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        # step -= warmup_steps
        # max_steps -= warmup_steps
        # q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # end_lr = base_lr * 0.001
        # lr = base_lr * q + end_lr * (1 - q)
        # print(f'Updated learning rate:{lr}')
    # optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    # optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr #* args.learning_rate_weights


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

if __name__ == '__main__':
    main()