import math
import os
import logging
from sampling import SimulatedAnnealing
from editor import RobertaEditor
from args import get_args
import numpy as np

import warnings
from utils import set_seed
import datetime
from dateutil import tz
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = get_args()

    set_seed(args.seed)
    editor = RobertaEditor(args)
    editor.to(device)
    sa = SimulatedAnnealing(args, editor).to(device)
    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    if args.task=='sentiment':
        with open('data/yelp/test.'+postfix, 'r', encoding='utf8') as f:
            data = f.readlines()
    else:
        with open('data/GYAFC/test_50.'+postfix, 'r', encoding='utf8') as f:
            data = f.readlines()

    batch_size = 1
    num_batches = math.ceil(len(data) / float(batch_size))

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    output_file = timestamp + '_' + args.task + '_' + 'seed=' + str(args.seed) + '_' + args.style_mode + '_' \
                  + str(args.style_weight) + '_' + args.direction + '.txt'
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
        T=0
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

            total_score_list = []
            cand_sent_list = []
            state_vec, pos_list = editor.state_vec(ref_olds)
            flag = 0
            for ops in [0,1,2]:
                seq_len = len(ref_oris[0].split())
                ops = np.array([ops])

                for positions in range(seq_len):
                    positions=[positions]
                    ref_news = editor.edit(ref_olds, ops, positions, sa.max_len)
                    accept_probs, index, ref_old_score, ref_new_score, old_style_score,new_style_scores, new_style_labels \
                        = sa.acceptance_prob(ref_news,ref_olds, ref_oris, T, ops, state_vec)

                    ref_hat = ref_news[index]
                    new_style_score=new_style_scores[index]
                    new_style_label=new_style_labels[index]

                    total_score_list.append(new_style_score)
                    cand_sent_list.append(ref_hat)

                    if args.early_stop == True:
                        if args.direction == '0-1' and new_style_label == 'positive':
                            print("Early Stopping!")
                            logging.info("Early Stopping!")
                            print("{} \ttotal score:{} {} style_score {} {}"
                                  .format(ref_hat, ref_old_score.item(),
                                          ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                            logging.info("{}\t total score:{} {}\t style_score {} {}"
                                         .format(ref_hat, ref_old_score.item(),
                                                 ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                            flag=1
                            break
                        elif args.direction == '1-0' and new_style_label == 'negative':
                            print("Early Stopping!")
                            logging.info("Early Stopping!")
                            print("{}\t total score:{} {} style_score {} {}"
                                  .format(ref_hat, ref_old_score.item(),
                                          ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                            logging.info("{}\t total score:{} {}\t style_score {} {}"
                                         .format(ref_hat, ref_old_score.item(),
                                                 ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                            flag = 1
                            break
                break

            if args.early_stop==True and flag==1:
                select_sent = cand_sent_list[-1]
            else:
                select_index=torch.argmax(torch.tensor(total_score_list).cuda())
                select_sent=cand_sent_list[select_index]
            logging.info('the selected sentence is {}'.format(select_sent))
            print('the selected sentence is {}'.format(select_sent))

            logging.info('\n')
            # return ref_hat
            f.write(select_sent + '\n')
            f.flush()

if __name__=="__main__":
    main()

