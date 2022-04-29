# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import GPT2Tokenizer,BartTokenizer

sys.path.append("")
from utils.dataset import SCIterator
from utils.textcnn import TextCNN

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = 'cuda' if cuda.is_available() else 'cpu'
special_tokens = [{'bos_token': '<bos>'},
                  {'eos_token': '<eos>'}, {'sep_token': '<sep>'},
                  {'pad_token': '<pad>'}, {'unk_token': '<unk>'}]
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# for x in special_tokens:
#     tokenizer.add_special_tokens(x)


def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('--gen_path', default='../data/gyafc/test.0', type=str, help='src')
    parser.add_argument('--direction', default='0-1', type=str, help='from 0 to 1')

    parser.add_argument('--max_len', default=30, type=int, help='max tokens in a batch')
    parser.add_argument('--embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('--dataset', default='gyafc', type=str, help='the name of dataset')
    parser.add_argument('--model', default='textcnn', type=str, help='the name of model')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('--batch_size', default=1, type=int, help='max sents in a batch')
    parser.add_argument("--dropout", default=0.5, type=float, help="Keep prob in dropout")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    test_src, test_tgt = [], []
    with open(opt.gen_path,'r') as f:
        for line in f.readlines():
            test_src.append(tokenizer.encode(line.strip())[:opt.max_len])
    max_num=len(test_src)
    print('[Info] {} instances from src test set'.format(len(test_src)))
    test_loader = SCIterator(test_src, test_tgt, opt, tokenizer.pad_token_id)

    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes,
                    num_filters, None, dropout=opt.dropout)
    model.to(device).eval()
    model.load_state_dict(torch.load('../checkpoints/textcnn_{}.chkpt'.format(
        opt.dataset)))

    total_num = max_num
    total_acc = 0.
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            x_batch, _ = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            print(F.softmax(logits, dim=-1))
            _, y_hat = torch.max(logits,dim=-1)

            if opt.direction=='0-1' and y_hat==1:
                total_acc+=1
            elif opt.direction=='1-0' and y_hat==0:
                total_acc+=1

    print(total_acc)
    print(total_num)

    print('Test: {}'.format('acc {:.4f}%').format(
        total_acc / total_num * 100))


if __name__ == '__main__':
    main()