#!/usr/bin/env python

import argparse
import os
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from dataset import get_datasets
from sklearn.metrics import auc, roc_auc_score
class Mapper(nn.Module):
    def __init__(self, prenet='resnet18', outdim=128):
        super(Mapper, self).__init__()

        model = getattr(models, prenet)(weights=None)

        self.model = list(model.children())[:-1]

        self.backbone = nn.Sequential(*self.model)

        if prenet == 'resnet18':
            nfc = 512
        elif prenet == 'resnet50':
            nfc = 2048

        self.fc1 = nn.Linear(nfc, outdim)
        # self.fc2 = nn.Linear(1024, outdim)

    def forward(self, x):
        bs = x.size(0)
        y = self.backbone(x)
        y = y.view(bs, -1)
        output = self.fc1(y)
        return output



device = 'cuda:0'
net_photo = Mapper()
net_photo = torch.nn.DataParallel(net_photo, device_ids=[0, 1]).to(device)
# address of the checkpoint
state = torch.load(
    '/home/n-lab/Amol/cl_quality/model_resnet18_100.pt')
net_photo.module.load_state_dict(state["net_photo"])


# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat-list', type=str,
                    default= '/media/n-lab/DATA1/Amol/fingerphoto/test/iitb/',#'/home/n-lab/Amol/contact-quality2/data/nbis_ridgebase/processed_fingerphotos/full/train/',
                    help='The feature list')
parser.add_argument('--pair-list', type=str,
                    default='/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/iitb',
                    help='whether the img in feature list is same person')
parser.add_argument('--quality-list', type=str,
                    default='/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_mod_ft_v8.2/converted_scores_cl_test.txt',
                    help='quality values')
parser.add_argument('--eval-type', type=str,
                    default='1v1',
                    help='The evaluation type')
parser.add_argument('--test-folds', type=int,
                    default=10,
                    help='')
parser.add_argument('--verifier', type=str, default='nbis')
parser.add_argument('--savename', type=str, default='')

def load_feat_pair(mode, args, verify):
    pairs = {}
    idx = 0
    quality_dict = {}
    dataset_name = 'iitb'
    if mode == 3:
        quality_list = f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/quality_new_v2_7_resnet50/scores_{dataset_name}.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vin_c10_mean_manual_labeling_v3_ft_ridgebase/scores.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_mod_ft_v8.2.5/converted_scores_ridgebase_train.txt'#
    elif mode == 1:
        # quality_list = '/home/n-lab/Amol/cl_quality/innovatrics/in_quality_biocop_test.txt'#'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nist_test/inn_nist_test_quality_scores.txt'#'/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/nbis_ridgebase/in_quality_full_ridgebase_train.txt' #
        quality_list = f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/ait_{dataset_name}_padded_normed_scores.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_nbis_v10/scores.txt'
    elif mode == 2:
        # quality_list = f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/in_{dataset_name}_quality.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vf_c10_v16/scores.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_vf_c10_v16/scores.txt'
        quality_list = f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/quality_new_v2_7/scores_{dataset_name}.txt'
    else:
        quality_list = f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/erc_evaluation/{dataset_name}/nfiq_quality_test_{dataset_name}_padded_normed_scores.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_mod_ft_v8.2.5/converted_scores_4670.txt'#'/home/n-lab/Amol/contact-quality2/results/lab2_resnet18_reg_innv2_v11/scores.txt'#'/home/n-lab/Amol/contact-quality2/data/iitb_polyu_biocop_quality/nfiq_iitb_polyu_biocop_full_test_scores.txt'#'/home/n-lab/Amol/contact-quality2/data/biocop_nist_final_patches/full/nfiq_biocop_nist_patches_full_test_w_splits_scores.txt'#'/home/n-lab/Amol/contact-quality2/data/nbis_ridgebase/nfiq_ridgebase_full_train_scores.txt'#

    with open(quality_list) as f:
        for record in f:
            # filename, score = record.strip('\n').split(' ')
            records = record.strip('\n').split(' ')
            filename = records[0]
            score = float(records[-1])
            quality_dict[filename.replace('wsq', 'png')] = int(score)
    if not verify:
        with open(args.pair_list + '/imp.txt') as f:
            for record in f:
                file_a, file_b = record.strip().split(' ')
                is_same = 0
                qlt = min(quality_dict[file_a], quality_dict[file_b])
                pairs[idx] = [file_a, file_b, is_same, qlt]
                idx += 1
        with open(args.pair_list + '/gen.txt') as f:
            for record in f:
                file_a, file_b = record.strip().split(' ')
                is_same = 1
                qlt = min(quality_dict[file_a], quality_dict[file_b])
                pairs[idx] = [file_a, file_b, is_same, qlt]
                idx += 1
    elif verify=='inn':
        score_dict = {}
        with open(args.pair_list + '/in_scores.txt') as f:
            for record in f:
                file_a, file_b, score, is_same = record.strip().split(' ')
                # is_same = 0
                qlt = min(quality_dict[file_a], quality_dict[file_b])
                pairs[idx] = [file_a, file_b, int(is_same), qlt] # int(is_same) for biocop, iitb, polyu, 1-int(is_same) for nbis_ridgebase
                score_dict[(file_a, file_b)] = float(score)
                idx += 1
        return pairs, score_dict

    elif verify=='vf':
        score_dict = {}
        with open(args.pair_list + '/vf_scores.txt') as f:
            for record in f:
                file_a, file_b, score, is_same = record.strip().split(' ')
                # is_same = 0
                qlt = min(quality_dict[file_a], quality_dict[file_b])
                pairs[idx] = [file_a, file_b, int(is_same), qlt] # int(is_same) for biocop, iitb, polyu, 1-int(is_same) for nbis_ridgebase
                score_dict[(file_a, file_b)] = float(score)
                idx += 1
        return pairs, score_dict

    elif verify=='nbis':
        score_dict = {}
        with open(args.pair_list + '/nbis_scores.txt') as f:
            for record in f:
                file_a, file_b, score, is_same = record.strip().split(' ')
                # is_same = 0
                qlt = min(quality_dict[file_a], quality_dict[file_b])
                pairs[idx] = [file_a, file_b, int(is_same), qlt] # int(is_same) for biocop, iitb, polyu, 1-int(is_same) for nbis_ridgebase
                score_dict[(file_a, file_b)] = float(score)
                idx += 1
        return pairs, score_dict

    return pairs


def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist


def calc_score(image_path, embeddings0, embeddings1, actual_issame, score_dict=None):
    # assert (embeddings0.shape[0] == embeddings1.shape[0])
    # assert (embeddings0.shape[1] == embeddings1.shape[1])
    dist = []
    if score_dict:
        for e0, e1 in zip(embeddings0.tolist(), embeddings1.tolist()):
            dist.append(score_dict[(''.join(e0), ''.join(e1))])
    else:
        dataloader = get_datasets(image_path, embeddings0.tolist(), embeddings1.tolist(), 1000)
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            im_1 = data[0]
            im_2 = data[1]

            im_1 = im_1.repeat(1, 3, 1, 1)
            im_2 = im_2.repeat(1, 3, 1, 1)

            net_photo.eval()
            with torch.no_grad():
                im_1 = im_1.to(device)
                im_2 = im_2.to(device)
                z_1 = net_photo(im_1)
                z_2 = net_photo(im_2)
                temp = ((z_1 - z_2) ** 2).sum(axis=1)
            dist = dist+temp.detach().tolist()
    # sort in a desending order
    dist = np.array(dist)
    pos_scores = np.sort(dist[actual_issame == 1])
    neg_scores = np.sort(dist[actual_issame == 0])
    if score_dict:
        pos_scores = np.flip(pos_scores)
        neg_scores = np.flip(neg_scores)
    return pos_scores, neg_scores


def save_pdf(fnmrs_list, unconsidered_rates, args):
    colors = ['red', 'blue', 'green', 'black']
    method_labels = ['A', 'B', 'C', 'D']
    plt.figure(figsize=(8, 4))
    plt.plot(unconsidered_rates[:len(fnmrs_list)], fnmrs_list, color=colors[0])
    plt.xlabel('Ratio of unconsidered image [%]')
    plt.ylabel('FNMR')
    plt.title('Testing on CFP, FMR=1e-3 (MagFace)')
    plt.savefig('test.png')
    plt.show()

def save_pdf2(fnmrs_list0, fnmrs_list1, fnmrs_list2, fnmrs_list3, unconsidered_rates, args):
    colors = ['red', 'blue', 'green', 'black']
    method_labels = ['A', 'B', 'C', 'D']
    fig = plt.figure(figsize=(16, 8))
    sorted_rejects = np.array(unconsidered_rates[:len(fnmrs_list2)]/100.)
    sorted_errors = np.array(sorted(fnmrs_list2))
    total_auc = auc(sorted_rejects, sorted_errors)
    fprs = sorted_rejects / sorted_rejects.max()
    tprs = 1 - sorted_errors / sorted_errors.max()
    p_auc = auc(fprs[fprs <= 0.25], tprs[fprs <= 0.25])
    p_auc2 = np.trapz(tprs[fprs <= 0.25], fprs[fprs <= 0.25])
    # print(sorted_rejects, sorted_errors)

    # print(auc(unconsidered_rates[:len(fnmrs_list0)], fnmrs_list0), auc(unconsidered_rates[:len(fnmrs_list1)], fnmrs_list1), auc(unconsidered_rates[:len(fnmrs_list2)], fnmrs_list2))
    plt.plot(unconsidered_rates[:len(fnmrs_list0)]/100., fnmrs_list0, color=colors[0], linestyle='dashdot', label='NFIQ2.0')#W/o Regional Quality
    plt.plot(unconsidered_rates[:len(fnmrs_list1)]/100., fnmrs_list1, color=colors[1], linestyle='dotted', label='AIT')#Innovatrics
    plt.plot(unconsidered_rates[:len(fnmrs_list2)]/100., fnmrs_list2, color=colors[2], linestyle='dashed', label='w/o fusion')#VeriFinger
    plt.plot(unconsidered_rates[:len(fnmrs_list3)]/100., fnmrs_list3, color=colors[3], linestyle='solid', label='Ours')  # VeriFinger
    # plt.plot(unconsidered_rates[:len(fnmrs_list3)], fnmrs_list3, color=colors[2], linestyle='solid', label='Combined')
    # plt.fill_between(sorted_rejects[fprs <= 0.25], tprs[fprs <= 0.25], np.zeros_like(tprs[fprs <= 0.25]),
    #                  alpha=0.2, color='blue', label= 'pAUC')
    plt.xlabel('Ratio of unconsidered image [%]')
    plt.ylabel('FNMR')
    plt.xlim([0, 0.4])
    # plt.ylim([0.0, 1.0])
    # plt.title('Testing on IITB')
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':24})
    if args.savename != '':
        plt.savefig(args.savename)

    # fig.set_facecolor('#EBEBEB')
    # # Style the grid.
    # fig.grid(which='major', color='white', linewidth=1.2)
    # fig.grid(which='minor', color='white', linewidth=0.6)
    # # Show the minor ticks and grid.
    # fig.minorticks_on()
    # # Now hide the minor ticks (but leave the gridlines).
    # fig.tick_params(which='minor', bottom=False, left=False)
    #
    # # Only show minor gridlines once in between major gridlines.
    # fig.xaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.yaxis.set_minor_locator(AutoMinorLocator(2))


    plt.show()

# f1 = open('./erc_unconsidered_biocop_m1_lq.txt', 'w')
# f2 = open('./erc_unconsidered_biocop_m2_lq.txt', 'w')
# f3 = open('./erc_unconsidered_biocop_m3_lq.txt', 'w')
# files = [f1, f2, f3]
def perform_1v1_quality_eval(mode, args, verify=None):
    # load features
    if not verify:
        feat_pairs = load_feat_pair(mode, args, verify)
    else:
        feat_pairs, score_dict = load_feat_pair(mode, args, verify)
    # ensemble feats
    embeddings0, embeddings1, targets, qlts = [], [], [], []
    pair_qlt_list = []  # store the min qlt
    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])
        qlt = v[3]

        # append
        embeddings0.append(feat_a)
        embeddings1.append(feat_b)
        targets.append(ab_is_same)
        qlts.append(qlt)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1, )
    qlts = np.array(qlts)
    qlts_sorted_idx = np.argsort(qlts)
    # print(embeddings0[qlts_sorted_idx[:10]], embeddings1[qlts_sorted_idx[:10]])
    num_pairs = len(targets)
    unconsidered_rates = np.arange(0, 0.98, 0.05)
    fnmrs_list = []
    for u_rate in unconsidered_rates:
        hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs):]
        # if np.round(u_rate, 2)==0.6:
        #     lq_pairs_idx = qlts_sorted_idx[:int(u_rate * num_pairs)]
            # for pair in zip(embeddings0[lq_pairs_idx].tolist(), embeddings1[lq_pairs_idx].tolist()):
                # print(f'{pair}')
                # files[mode].write(f"{''.join(pair[0])}, {''.join(pair[1])}\n")
        # print(embeddings0[hq_pairs_idx], embeddings1[hq_pairs_idx])
        if not verify:
            pos_dists, neg_dists = calc_score(args.feat_list, embeddings0[hq_pairs_idx],
                                          embeddings1[hq_pairs_idx],
                                          targets[hq_pairs_idx])
        else:
            pos_dists, neg_dists = calc_score(args.feat_list, embeddings0[hq_pairs_idx],
                                              embeddings1[hq_pairs_idx],
                                              targets[hq_pairs_idx], score_dict)

        fmr = 1
        idx = len(neg_dists) - 1
        num_query = len(pos_dists)
        while idx >= 0:
            thresh = neg_dists[idx]
            if not verify:
                num_acc = sum(pos_dists < thresh)
            else:
                num_acc = sum(pos_dists > thresh)
            fnmr = 1.0 * (num_query - num_acc) / num_query
            # print(f'total_pairs:{len(pos_dists)+len(neg_dists)}, num_query:{num_query}, num_acc:{num_acc}, neg_pairs:{len(neg_dists)}')
            # print(num_query, num_acc)
            if fmr == 1e-3:
                sym = ' ' if thresh >= 0 else ''
                line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
                # print(line)
                fnmrs_list.append(fnmr)

            if idx == 0:
                break
            idx /= 10
            idx = int(idx)
            fmr /= float(10)
    print(fnmrs_list)
    # np.save(args.quality_list.replace('.list', '.npy'), np.array(fnmrs_list))
    # save_pdf(fnmrs_list, 100 * unconsidered_rates, args)
    return fnmrs_list, 100*unconsidered_rates


def main():
    args = parser.parse_args()
    # perform_1v1_quality_eval(args)
    verifier = args.verifier
    fnmr_list3= None
    fnmr_list0, unconsidered_rates_percent = perform_1v1_quality_eval(0, args, verify=verifier)
    fnmr_list1, unconsidered_rates_percent = perform_1v1_quality_eval(1, args, verify=verifier)
    fnmr_list2, unconsidered_rates_percent = perform_1v1_quality_eval(2, args, verify=verifier)
    fnmr_list3, unconsidered_rates_percent = perform_1v1_quality_eval(3, args, verify=verifier)
    # unconsidered_rates_percent = np.arange(0, 0.98, 0.05)*100
    # fnmr_list0 = [0.6916666666666667, 0.686372121966397, 0.6831489915419648, 0.6770616770616771, 0.6763686763686764, 0.6739766081871345, 0.6690196078431373, 0.656414762741652, 0.6590106007067138, 0.6567717996289425, 0.6492248062015504, 0.6458333333333334, 0.6354401805869074, 0.6263736263736264, 0.6091794158553546, 0.5961227786752827, 0.592057761732852, 0.5879732739420935, 0.5623003194888179, 0.5613207547169812] # PolyU NFIQ:
    # fnmr_list1 = [0.6916666666666667, 0.6922619047619047, 0.6922619047619047, 0.6922619047619047, 0.6916666666666667, 0.6928571428571428, 0.6928571428571428, 0.6996587030716723, 0.7152917505030181, 0.7130801687763713, 0.7120253164556962, 0.7146118721461188, 0.7113772455089821, 0.7118193891102258, 0.7107692307692308, 0.7198581560283688, 0.7112970711297071, 0.7037037037037037, 0.6782006920415224, 0.6666666666666666] # PolyU Pred:
    # fnmr_list1 = [0.9, 0.8998330550918197, 0.8998330550918197, 0.9015025041736227, 0.9031719532554258, 0.9015025041736227, 0.9015025041736227, 0.9015025041736227, 0.8998330550918197, 0.8909774436090225, 0.8947368421052632, 0.8834586466165414, 0.8834586466165414, 0.8834586466165414, 0.9009009009009009, 0.9081081081081082, 0.9241379310344827, 0.9259259259259259, 0.9, 0.8648648648648649] # IITB Pred:
    # fnmr_list0 = [0.9, 0.8976109215017065, 0.8957169459962756, 0.8959537572254336, 0.8992094861660079, 0.8956158663883089, 0.8964059196617337, 0.8961625282167043, 0.8946135831381733, 0.8908629441624365, 0.8970189701897019, 0.9020771513353115, 0.9, 0.8964285714285715, 0.9018867924528302, 0.8898305084745762, 0.8787878787878788, 0.8707482993197279, 0.853448275862069, 0.8285714285714286] # IITB NFIQ:
    # fnmr_list0 = [0.7540453074433657, 0.7448979591836735, 0.7398081534772182, 0.7358490566037735, 0.7286096256684492, 0.7117903930131004, 0.6962616822429907, 0.6845637583892618, 0.6748251748251748, 0.6519230769230769, 0.6402569593147751, 0.6264501160092807, 0.6318407960199005, 0.6072423398328691, 0.6136363636363636, 0.6, 0.5814977973568282, 0.6077348066298343, 0.6062992125984252, 0.5909090909090909] # BIOCOP NFIQ:
    # fnmr_list1 = [0.7540453074433657, 0.755939524838013, 0.7580993520518359, 0.7580993520518359, 0.7591792656587473, 0.7602591792656588, 0.7624190064794817, 0.7667386609071274, 0.7732181425485961, 0.7732181425485961, 0.7786177105831533, 0.7239488117001828, 0.6765498652291105, 0.6637426900584795, 0.6778523489932886, 0.6680161943319838, 0.6631016042780749, 0.6878980891719745, 0.6470588235294118, 0.5660377358490566] #BIOCOP PRED:
    save_pdf2(fnmr_list0, fnmr_list1, fnmr_list2, fnmr_list3, unconsidered_rates_percent, args)

    # print()


if __name__ == '__main__':
    main()


#############################ft_v8.2.1#####################################

#IITB: [0.9, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.9015025041736227, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.8998330550918197, 0.9375, 0.9453125, 0.9375, 0.9453125, 0.9375, 0.9302325581395349, 0.813953488372093]