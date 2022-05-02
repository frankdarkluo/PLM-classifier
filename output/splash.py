with open('gpt3-curie-001_500_1-0.txt', 'r', encoding='utf8') as f, \
    open('gpt3-curie-001_1-0.txt','w',encoding='utf8') as of:
        datas=f.readlines()
        for idx, data in enumerate(datas):
            if 'Generated:' in datas[idx]:
                out_sent=datas[idx+1].strip().split('}')[0].lower()
                of.write(out_sent+'\n')