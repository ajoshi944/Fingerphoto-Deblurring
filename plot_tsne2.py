import numpy as np
import argparse
from sklearn.linear_model import Ridge, RidgeCV
import pickle
from sklearn.manifold import TSNE
# from tsnecuda import TSNE as tsneC
# from openTSNE import TSNE as tsneO
import resnet18_dual_w_SA as models_mlp
from dataset_dual import get_datasets
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def main(args):
    exp_name = args.exp_name
    scores = []

    pre_feat = False
    if pre_feat:
        quality_feat = np.load('/media/n-lab/DATA1/Amol/ReIQA-main/feats_content_aware/iitb_content_aware_features.npy')
        content_feat = np.load('/media/n-lab/DATA1/Amol/ReIQA-main/feats_content_aware/iitb_content_aware_features.npy')
        quality_feat = quality_feat.reshape(quality_feat.shape[0], -1)
        content_feat = content_feat.reshape(content_feat.shape[0], -1)
        feats = np.concatenate([quality_feat, content_feat], axis=1)
        with open('/media/n-lab/DATA1/Amol/ReIQA-main/test/reiqa_iitb_normed_scores.txt', 'r') as f:
            for record in f:
                _, score = record.strip().split(' ')
                scores.append(score)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        test_dataset, test_dataloader, dataset_name = get_datasets(clf=False, train=False, weights=False, batch_size=1)
        model = models_mlp.updated_resnet18()
        # qmodel = models_mlp.Quality()
        ckpt = torch.load(
            f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/{args.exp_name}/latest_model.pth')
        model.load_state_dict(ckpt['model'])
        # qmodel.load_state_dict(ckpt['quality_model'])
        model = model.to(device)
        # qmodel = qmodel.to(device)
        model.eval()
        # qmodel.eval()
        feats = []
        prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size))
        for i, (candidate, filename, cand_nfiq) in prog_bar:
            # if i == 100:
            #     break
            with torch.no_grad():
                candidate = candidate.to(device)
                feat = model(candidate, pool=True)
                # qscore, _ = qmodel(feat)
                feats.append(feat.squeeze(3).squeeze(2).cpu().numpy())
                scores.append(cand_nfiq.numpy())#qscore.cpu().numpy()
            # print(scores.shape, feats.shape)
            # exit()
        feats = np.concatenate(feats)  # np.asarray(feats)[:10]
    scores = np.asarray(scores)#np.asarray(scores)[:10]
    score_bins = np.digitize(scores, bins=[40])#
    # X = np.random.rand(10, 512)
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    z = tsne.fit_transform(feats)
    # print(z.shape)
    # exit()
    plt.subplots(figsize=(10, 8))
    colors = ["salmon", "cornflowerblue", "seagreen", "aqua", "khaki"]
    labels = ["0-40", "41-100"]#["0-20", "21-40", "41-60", "61-80", "81-100"] ["0-10", "11-40", "41-60", "61-100"]
    for i, l in enumerate(labels):
        # if l in (proto_target, "prototype"):
        #     c, m, a, ec = "k", "X", 1.0, None
        # else:
        c, m, a, ec = colors[i], "o", 0.5, "w"
        # print(z[np.where(score_bins == i), :])
        x, y = z[np.where(score_bins == i), :].T
        plt.scatter(x, y, 50, c=c, marker=m, alpha=a, edgecolors=ec, label=l)

    plt.axis("off")
    # if legends:
    plt.legend(loc='lower right', prop={'size':18})
    plt.show()
    # if title is not None:
    #     plt.title(title)
    # plt.savefig(filename)
    # plt.close("all")
def parse_args():
    parser = argparse.ArgumentParser(description="linear regressor")
    # parser.add_argument('--feat_path', type=str, help='path to features file')
    # parser.add_argument('--ground_truth_path', type=str, \
    #                     help='path to ground truth scores')
    parser.add_argument('--alpha', type=float, default=0.1, \
                        help='regularization coefficient')
    parser.add_argument('--exp_name', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)