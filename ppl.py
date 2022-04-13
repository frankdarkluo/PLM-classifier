from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# with open('data/yelp/dualrl.txt','r',encoding='utf8') as f:
#     data=[line.strip() for line in f.readlines()]
#     encodings = tokenizer("\n\n".join(data), return_tensors="pt")

import torch
from editor import RobertaEditor
from args import get_args
max_length = model.config.n_positions
stride = 1024

args=get_args()
rbt_editor=RobertaEditor(args)

perpl=[]
with open(args.outfile,'r',encoding='utf8') as f:
    datas=f.readlines()
    for data in datas:
        _, gpt_tokens=rbt_editor.plm_token([data])
        #input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(gpt_token) for gpt_token in gpt_tokens]).to(device)
        encodings = tokenizer(data.strip(), return_tensors="pt")
        input_ids=encodings.input_ids
        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        perpl.append(ppl.cpu().numpy())

print(np.mean(perpl))