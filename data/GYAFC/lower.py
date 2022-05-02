from nltk import word_tokenize
def lower_down(prefix,style):
    with open(prefix+'.'+style,'r',encoding='utf8') as f, open(prefix+'n.'+style,'w',encoding='utf8') as of:
        datas=f.readlines()
        for data in datas:
            line=data.strip().lower()
            of.write(line+'\n')

def select(infile,outfile=None):
    with open(infile,'r',encoding='utf8') as f, open(outfile,'w',encoding='utf8') as of:
        datas=f.readlines()
        for data in datas:
            real_data = data.strip()
            data_length=len(word_tokenize(real_data))
            if data_length<20:
                of.write(real_data+'\n')

# select('test.1','test_select.1')
# select('test.0','test_select.0')







