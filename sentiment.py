from transformers import pipeline
import torch
import math
import argparse

#classifier = pipeline("sentiment-analysis")
classifier = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
#classifier = pipeline(model="EleutherAI/gpt-neo-1.3B")

pos=0
neg=0

parser = argparse.ArgumentParser()
parser.add_argument('--outfile', default='results/try1/2022-03-27_03:18:22_pipeline_3_1-0.txt', type=str)
args=parser.parse_args()

with open(args.outfile,'r',encoding='utf8') as f:
    datas=f.readlines()

    for idx,line in enumerate(datas):
        if idx< 500:
            res=classifier(line.strip())
            if res[0]['label'].lower()=='negative':
                pos+=1

            else:neg+=1
        # else:
        #     res = classifier(line.strip())
        #     if res[0]['label'].lower() == 'negative':
        #         pos += 1
        #     else:
        #         neg += 1
    #
    length=idx+1 # 500

print("POS ACC is {}".format(pos/length))


# data_test1=[46.8,24.2,1/166]  # 定义测试数据
# data_test2=[81.3, 47.6, 1/345]  # 定义测试数据
# data_test3=[73.7, 40.6, 1/184]  # 定义测试数据
#
# def geometric_mean(data):  # 计算几何平均数
#     total=1
#     for i in data:
#         total*=i #等同于total=total*i
#     return pow(total,1/len(data))
#
# print(geometric_mean(data_test1))
# print(geometric_mean(data_test2))
# print(geometric_mean(data_test3))

