import os
import sys
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import argparse
import csv
BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]

def load_data(file):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        datas = of.readlines()#[:100]
        for idx, data in enumerate(datas):
            strs.append(data.strip())

    str_list = [seq.strip().lower().split() for seq in strs]

    return str_list

def load_ref_data(ref_path):
    refs=[[]]*50

    for file in os.listdir(ref_path):
        with open(ref_path+file,'r',encoding='utf8') as f:
            lines=f.readlines()
            for j, line in enumerate(lines):
                line = line.strip().lower().split()
                temp=refs[j].copy()
                temp.append(line.copy())
                refs[j]=temp.copy()
    return refs

def metric(args):

    infer =load_data(args.gen_path)
    golden=load_ref_data(args.ref_path)

    # eval bleu
    sf=SmoothingFunction()
    corp_model_bleu1 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[0])
    corp_model_bleu2 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[1])
    corp_model_bleu3 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[2])
    corp_model_bleu4 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[3])

    nltk_bleu=[corp_model_bleu1, corp_model_bleu2, corp_model_bleu3, corp_model_bleu4]
    print('BLEU', nltk_bleu)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', default='data/yelp/pos2neg_50references/', type=str)
    parser.add_argument('--gen_path', default='BLEU/pos2neg/generate_pos2neg.txt', type=str)
    # parser.add_argument('--ori_path', default='data/yelp/test.0', type=str)
    parser.add_argument("--task", type=str, default='zero-shot')
    args = parser.parse_args()
    metric(args)
