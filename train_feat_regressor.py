import numpy as np
import argparse
from sklearn.linear_model import Ridge, RidgeCV
import pickle
import resnet18_dual_w_SA as models_mlp
from dataset_dual import get_datasets
import torch
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def main(args):
    exp_name = args.exp_name
    train_dataset, train_dataloader = get_datasets(clf=False, train=True, weights=False, batch_size=1)
    model = models_mlp.updated_resnet18(args)
    model.load_state_dict(torch.load(f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/{args.exp_name}/latest_model.pth')['model'])
    model = model.to(device)
    model.eval()
    scores = []
    feats = []
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, data_dict in prog_bar:
        # if i == 100:
        #     break
        with torch.no_grad():
            candidate = data_dict['candidate'].to(device)
            feats.append(model(candidate).squeeze().cpu().numpy())
        scores.append(data_dict['cand_innq'].numpy())
    # print(scores.shape, feats.shape)
    # exit()
    scores = np.asarray(scores)
    feat = np.asarray(feats)
    print(scores.shape, feat.shape)
    # train regression
    alphas = [0.001, 0.01, args.alpha, 1, 10]
    reg = RidgeCV(alphas=alphas).fit(feat, scores)
    print("The train score for ridge model is {}".format(reg.score(feat, scores)))
    pickle.dump(reg, open(f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/results/{exp_name}/lin_regressor.save', 'wb'))


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