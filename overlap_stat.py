import os
import sys
from nltk.translate.bleu_score import corpus_bleu
import statistics
import argparse

def load_data(file):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        datas = of.readlines()#[:100]
        for idx, data in enumerate(datas):
            data=data.strip().lower()
            strs.append(data)

    str_list = [seq.split() for seq in strs]

    return str_list

def load_ref_data(ref_path,N=50):
    refs=[[]]*N

    for file in os.listdir(ref_path):
        with open(ref_path+file,'r',encoding='utf8') as f:
            lines=f.readlines()[:N]
            for j, line in enumerate(lines):
                line = line.strip().lower()
                line=line.split()
                temp=refs[j].copy()
                temp.append(line.copy())
                refs[j]=temp.copy()
    return refs

def check_diff(infer,gold):
    num_list = []
    for idx,sent in enumerate(infer):
        gold_list=gold[idx]
        num=0
        for gold_sent in gold_list:
            num+=len([token for token in gold_sent if token not in sent])
        num_list.append(num)

    avg=statistics.mean(num_list)/4
    print("avg differne is {} tokens".format(avg))




def metric(args):
    infer =load_data(args.gen_path)
    ref_path='./data/{}/{}_ref/'.format(args.dataset,args.task)
    gold=load_ref_data(ref_path,args.N)
    check_diff(infer,gold)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ref_path', default='data/gyafc/pos2neg_ref/', type=str)
    parser.add_argument('--dataset',default='gyafc_500',type=str)
    parser.add_argument('--task', default='pos2neg', type=str)
    parser.add_argument('--gen_path', default='data/gyafc_500/test.1', type=str)
    parser.add_argument("--N",default=500,type=int)
    args = parser.parse_args()
    metric(args)
