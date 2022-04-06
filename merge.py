import os

dir='data/yelp/references/'
with open(dir+'reference3','w',encoding='utf8') as of, \
    open(dir+'reference3.0','r',encoding='utf8') as f1,open(dir+'reference3.1','r',encoding='utf8') as f2:
    data1=f1.readlines()
    data2=f2.readlines()
    for data in data1:
        of.write(data)
    for data in data2:
        of.write(data)