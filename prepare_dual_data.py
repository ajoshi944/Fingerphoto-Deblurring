import os
import random
from glob import glob
from tqdm import tqdm
import numpy as np

def dump_dict(dict, file, mode, label=None):
    f = open(file, mode)
    for key, val in dict.items():
        f.write(f"{' '.join(key)} {val[0]} {val[1]} {val[2]} {label if label else ''}\n")

def get_index(start, end, exclude):
    choice_list = list(range(start, exclude)) + list(range(exclude+1, end))
    return random.choice(choice_list)

def get_records(x, names):
    lno = 0
    records = []
    for record in x:
        if names:
            _, _, s = record.strip('\n').split(' ')
            records.append(s)
        else:
            records = record.split(',')[:-1]
        lno += 1
    return records
def matching_scores(probe_dir, gal_dir, innv_file, vf_file, nbis_file):
    innvf = open(innv_file, 'r')
    vff = open(vf_file, 'r')
    nbisf = open(nbis_file, 'r')
    probe_list = sorted(glob(probe_dir + '/*.png'))
    gal_list = sorted(glob(gal_dir + '/*.png'))
    gen_score = None
    cnt = 0
    score_gen = {}
    score_imp = {}
    records_innv = get_records(innvf, False)
    records_vf = get_records(vff, False)
    records_nbis = get_records(nbisf, True)
    # print(lno)
    # sc_min = np.asarray(records, dtype=int).min()
    # sc_max = np.asarray(records, dtype=int).max()
    print(f'Expected scores: {len(probe_list)**2}, Innv scores:{len(records_innv)}, Vf scores:{len(records_vf)}, Nbis scores:{len(records_nbis)}')# min: {sc_min}, max: {sc_max}
    # exit()
    # normalize = lambda x: np.round((x - sc_min) / (sc_max - sc_min) , decimals=4)
    innv_records = [records_innv[i:i + len(probe_list)] for i in range(0, len(records_innv), len(probe_list))]
    vf_records = [records_vf[i:i + len(probe_list)] for i in range(0, len(records_vf), len(probe_list))]
    nbis_records = [records_nbis[i:i + len(probe_list)] for i in range(0, len(records_nbis), len(probe_list))]
    for idx, scores in tqdm(enumerate(innv_records), total=len(innv_records)):
        # print(f'Image processing:{os.path.basename(img_list[idx])}')
        score_len = len(scores)
        chosen_j = get_index(0, score_len, idx)
        vf_scores = vf_records[idx]
        nbis_scores = nbis_records[idx]
        for j, score in enumerate(scores):
            if j == idx:
                gen_innv_score = int(score)
                gen_vf_score = int(vf_scores[j])
                gen_nbis_score = int(nbis_scores[j])
                score_gen[(os.path.basename(probe_list[idx]), os.path.basename(gal_list[j]))] = [gen_innv_score, gen_vf_score, gen_nbis_score]
            else:
                # if gen_score == int(score):
                #     print(os.path.basename(img_list[j]))
                #     cnt += 1
                # if os.path.basename(probe_list[idx]) not in score_imp.keys():
                if j == chosen_j:
                    imp_innv_score = int(score)
                    imp_vf_score = int(vf_scores[j])
                    imp_nbis_score = int(nbis_scores[j])
                    score_imp[(os.path.basename(probe_list[idx]), os.path.basename(gal_list[j]))] = [imp_innv_score, imp_vf_score, imp_nbis_score]
                # else:
                #     self.score_imp[os.path.basename(img_list[idx])].append(int(score))
    innvf.close()
    vff.close()
    nbisf.close()
    print(len(score_gen), len(score_imp))
    return score_gen, score_imp
# dataset_name = 'rset6'
# score_gen, score_imp = matching_scores(f'/home/n-lab/Amol/fingerphoto_quality/{dataset_name}/gallery',
#                 f'/home/n-lab/Amol/fingerphoto_quality/{dataset_name}/probe',
#                 f'/home/n-lab/Amol/fingerphoto_quality/innovatrics/in_scores_{dataset_name}.csv',
#                 f'/home/n-lab/Amol/fingerphoto_quality/verifinger/vf_scores_{dataset_name}.csv',
#                 f'/home/n-lab/Amol/fingerphoto_quality/nbis/nbis_scores_{dataset_name}.txt')
# new_min = np.asarray(list(score_gen.values())).min()
# new_max = np.asarray(list(score_gen.values())).max()
# print(new_min, new_max)
# dump_dict(score_gen, f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/all_scores_{dataset_name}.txt', 'w', '0')
# dump_dict(score_imp, f'/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/all_scores_{dataset_name}.txt', 'a', '1')

#######################################################################################################################
def add_quality(scores_file, quality_file, nfiq_file, innq_file, target_file):
    ms = open(scores_file, 'r')
    qs = open(quality_file, 'r')
    ns = open(nfiq_file, 'r')
    ins = open(innq_file, 'r')
    ts = open(target_file, 'a')

    quality_dict = {}
    nfiq_dict = {}
    innq_dict = {}
    record_dict = {}
    for quality_scores in qs:
        name, score = quality_scores.strip('\n').split(' ')
        quality_dict[name] = score
    qs.close()
    for nfiq_scores in ns:
        name, score = nfiq_scores.strip('\n').split(' ')
        nfiq_dict[name] = score
    ns.close()
    for innq_scores in ins:
        name, score = innq_scores.strip('\n').split(' ')
        innq_dict[name] = score
    ins.close()
    list_s = []
    for record in ms:
        queryname, nm1, s, lb = record.strip('\n').split(' ')
        record_dict[(queryname, nm1)] = (s, lb)
        list_s.append(s)
    ms.close()
    arr_s = np.asarray(list_s, dtype=int)
    s_min = arr_s.min()
    s_max = arr_s.max()
    arr_s = (arr_s - s_min) / (s_max - s_min)
    cnt = 0
    for idx, (k, v) in enumerate(list(record_dict.items())):
        queryname, nm1 = k
        _, lb = v
        new_record = f"{queryname} {nm1} {np.round(arr_s[idx], decimals=4)} {lb} {nfiq_dict[queryname]} {nfiq_dict[nm1]} {quality_dict[queryname]} {innq_dict[queryname]} {innq_dict[nm1]}\n"
        ts.write(new_record)
        cnt += 1
    ts.close()
    print(f'Transferred:{cnt}')

# add_quality('/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/in_scores_comb_set2.txt',
#             '/home/n-lab/Amol/fingerphoto_quality/labeling/innovatrics/quality_in_all_set_custom_splits.txt',
#             '/home/n-lab/Amol/fingerphoto_quality/nfiq_all_set_scores.txt',
#             '/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/in_qual_all_set.txt',
#             '/home/n-lab/Amol/fingerphoto_quality/contact_quality2/data/in_data_all_set.txt')
#######################################################################################################################
