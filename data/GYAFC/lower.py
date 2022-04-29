
def lower_down(prefix,style):
    with open(prefix+'.'+style,'r',encoding='utf8') as f, open(prefix+'n.'+style,'w',encoding='utf8') as of:
        datas=f.readlines()
        for data in datas:
            line=data.strip().lower()
            of.write(line+'\n')



