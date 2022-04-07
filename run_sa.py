import json
import math
import os.path
import os
import logging
from sampling import SimulatedAnnealing
from editor import RobertaEditor
from args import get_args
import numpy as np
import random
import warnings

import datetime
from dateutil import tz
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
import torch
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = get_args()
    editor = RobertaEditor(args)
    editor.to(device)
    sa = SimulatedAnnealing(args, editor, args.t_init, args.C, args.fluency_weight, args.keyword_weight,
                            args.sent_weight,args.style_weight, args.max_steps).to(device)

    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    if args.task=='sentiment':
        with open('data/yelp/test_50.'+postfix, 'r', encoding='utf8') as f:
            data = f.readlines()
    else:
        with open('data/GYAFC/test_50.'+postfix, 'r', encoding='utf8') as f:
            data = f.readlines()

    batch_size = 1
    num_batches = math.ceil(len(data) / float(batch_size))

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    output_file = timestamp+'_'+args.task+'_'+args.style_mode + '_' + str(args.style_weight) + '_' + args.direction + '.txt'
    log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',
                        filename=log_txt_path,
                        filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    word_pairs ={"ca n't": "can not","n't":"not"}

    logging.info(args)

    with open(of_dir + output_file, 'w', encoding='utf8') as f:
        for i in range(num_batches):

            batch_data = data[batch_size * i:batch_size * (i + 1)]
            input_batch = [batch_data]
            ref_olds = []

            for input in input_batch[0]:
                temp = input.strip().lower()
                for k, v in word_pairs.items():
                    temp = temp.replace(k, v)
                line = temp
                ref_olds.append(line)
            #state_vec=None # ablation study

            batch_size = len(ref_olds)  # 1
            ref_oris=ref_olds

            for t in range(sa.max_steps):
                T = max(sa.t_init - sa.C * t, 0)

                #ablation study
                # num=random.random()
                # if num>=0.2:
                #     ops = np.array([1])
                # elif 0.2>num>=0.1:
                #     ops = np.array([0])
                # elif num<0.1:
                #     ops=np.array([2])
                if args.action=="all":
                    ops = np.random.randint(0, 3, batch_size)
                elif args.action=="insert": ops=np.array([0])
                elif args.action=="replace":ops=np.array([1])
                elif args.action=="delete": ops=np.array([2])

                positions = [random.randint(0, len(i.split()) - 1) for i in ref_olds]

                if args.semantic_mode != 'sent':
                    state_vec, pos_list = editor.state_vec(ref_olds)

                #keep nouns unchangeable
                if args.keyword_pos==True:
                    if args.semantic_mode != 'sent' and ops!=0:# ==replace or delete
                        while pos_list[0][positions[0]] in ['NN', 'NNS','NNP','NNPS']:
                            positions = [random.randint(0, len(i.split()) - 1) for i in ref_olds]

                ref_news = editor.edit(ref_olds, ops, positions, sa.max_len)
                accept_probs, index, ref_old_score, ref_new_score, old_style_score,new_style_score, new_style_label \
                    = sa.acceptance_prob(ref_news,ref_olds, ref_oris, T, ops, state_vec)

                ref_hat = ref_news[index]
                new_style_score=new_style_score[index]
                new_style_label=new_style_label[index]

                if batch_size == 1:
                    accept_prob = accept_probs[0]

                # for idx, accept_prob in enumerate(accept_probs):
                if sa.choose_action([accept_prob, 1 - accept_prob]) == 0:
                # if accept_prob == 1:
                    print("A is {}, T is {}:\t{} total score:{} {} style_score {} {}"
                          .format(accept_prob, T, ref_hat, ref_old_score.item(),
                                  ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                    logging.info("A is {}, T is {}:\t{}\ttotal score:{} {}\t style_score {} {}"
                                .format(accept_prob, T, ref_hat, ref_old_score.item(),
                                        ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                    ref_olds = [ref_hat]

                if args.early_stop==True:
                    if args.direction=='0-1' and new_style_label=='positive':
                        print("Early Stopping!")
                        break
                    elif args.direction=='1-0' and new_style_label=='negative':
                        print("Early Stopping!")
                        break

            logging.info('\n')
            # return ref_olds
            for sa_output in ref_olds:
                f.write(sa_output + '\n')
                f.flush()

if __name__=="__main__":
    main()

