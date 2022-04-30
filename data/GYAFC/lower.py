
def lower_down(prefix,style):
    with open(prefix+'.'+style,'r',encoding='utf8') as f, open(prefix+'n.'+style,'w',encoding='utf8') as of:
        datas=f.readlines()
        for data in datas:
            line=data.strip().lower()
            of.write(line+'\n')

def sssplit(infile,outfile=None):
    with open(infile,'r',encoding='utf8') as f, open('test_tok.0','w',encoding='utf8') as of:
        datas=f.readlines()
        for data in datas:
            real_data = data.split('\t')[0]
            of.write(real_data+'\n')

sssplit('dualrl.0')





