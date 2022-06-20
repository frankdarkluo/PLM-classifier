import os
import argparse
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('edit distance')
parser.add_argument('--steps', default='3', type=str, help='editing steps')
parser.add_argument('--seed', default=42, type=int, help='pseudo random number seed')
parser.add_argument('--dataset', default='yelp', type=str, help='dataset')

opt = parser.parse_args()
torch.manual_seed(opt.seed)

path = 'output/SAHC_{}_output/{}_steps/'.format(opt.dataset, opt.steps)

def main():
    file1=os.path.join(path,'zero-shot_0-1.log')
    file2=os.path.join(path,'zero-shot_1-0.log')

    with open(file1,'r',encoding='utf8') as f1, open(file2,'r',encoding='utf8') as f2:
        data1=f1.readlines()
        data2=f2.readlines()
        edit_num1, sent_num1, edit_dic1=count_steps(data1)
        edit_num2, sent_num2, edit_dic2=count_steps(data2)

        def dict_union(A,B):
            C = {}
            for key in A:
                if B.get(key):
                    C[key] = A[key] + B[key]
                else:
                    C[key] = A[key]
            for key in B:
                if not A.get(key):
                    C[key] = B[key]
            return C

    edit_hist_dic=dict_union(edit_dic1,edit_dic2)

    draw_histo(edit_hist_dic)

    sent_num=sent_num1+sent_num2
    edit_num=edit_num1+edit_num2

    print("for 0-1.txt, editing on {} sentences needs {} steps in total, {} steps in average".
          format(sent_num1,edit_num1,edit_num1/sent_num1))
    print("for 1-0.txt, editing on {} sentences needs {} steps in total, {} steps in average".
          format(sent_num2, edit_num2, edit_num2 / sent_num2))
    print("for two texts as a whole, editing on {} sentences needs {} steps in total, {} steps in average".
          format(sent_num, edit_num, edit_num / sent_num))


def count_steps(datas):
    sent_num=0
    edit_num=0
    edit_dic={"1":[],"2":[],"3":[],"4":[],"5":[]}
    for data in datas:
        data=data.strip().lower()
        if ' steps, the selected sentence is' in data:
            sent_num+=1
            number=data.split(' steps, the selected sentence is')[0][-1]
            edit_dic[str(number)].append(1)
            edit_num+=int(number)

    edit_hist_dic = {key: sum(value) for key, value in edit_dic.items()}

    return edit_num,sent_num,edit_hist_dic

def draw_histo(edit_hist_dic,heng=0):
    edit_hist_value = edit_hist_dic.items()
    x = []
    y = []
    for d in edit_hist_value:
        x.append(d[0])
        y.append(d[1])
    if heng == 0:
        edit_bar=plt.bar(x, y,0.5,edgecolor='grey')
        plt.title('Edit Distance')
        plt.xlabel('Editing steps', fontsize=15)
        plt.bar_label(
            edit_bar,
            labels=None,
            fmt='%g',
            label_type='center',
            padding=30,
        )
        plt.ylabel('# of Samples', fontsize=15)
        plt.savefig(os.path.join(path,'{}step_distance.png'.format(opt.steps)), bbox_inches='tight')
        plt.show()
        plt.close()
        return
    elif heng == 1:
        plt.barh(x, y)
        plt.show()
        return
    else:
        return "heng is only 0 or 1!"


if __name__ == '__main__':
    main()