
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
model_dir = 'roberta-large'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForMaskedLM.from_pretrained(model_dir, return_dict=True).to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_dir)

max_len=0
total_len=0
with open('./data/gyafc/test.0','r',encoding='utf8') as f,open('./data/gyafc/test_tok.0','w',encoding='utf8') as of:
    datas=f.readlines()
    for data in datas:
        data=data.strip().lower()
        tokens=tokenizer.tokenize(data)
        tokens=[token.split('Ġ')[-1] for token in tokens]
        total_len+=len(tokens)
#         if len(tokens)>max_len:
#             max_len=len(tokens)
#             print(tokens)
#
# print(max_len)
# print(total_len/len(datas))
