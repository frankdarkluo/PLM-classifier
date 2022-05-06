from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering, \
    GPTNeoForCausalLM, GPT2Tokenizer,GPTJForCausalLM
import torch
from torch import nn
import numpy as np
from utils.functions import softmax

topk=1000
model_name="EleutherAI/gpt-neo-1.3B"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model =GPTNeoForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model =GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
# model=GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",revision = "float16",
#                                       torch_dtype = torch.float16,low_cpu_mem_usage = True).cuda()
prefix="""
Sentence: This movie is very nice.
Sentiment: {positive}

#####

Sentence: I hated this movie, it sucks.
Sentiment: {negative}

#####

Sentence: This movie was actually pretty funny.
Sentiment: {positive}

#####

"""
line="ever since joes has changed hands it is just gotten worse and worse ."
input="Sentence: "+str(line)+"\n"
postfix = "Sentiment: {"
seq=prefix+input+postfix

indexed_tokens = tokenizer.encode(seq)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens]).cuda()
# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Get the predicted next sub-word
# if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
probs = predictions[0, -1, :]

pos_logits=probs[tokenizer.encode('positive')]
neg_logits=probs[tokenizer.encode('negative')]
emo_logits=torch.concat([pos_logits,neg_logits])
softmax_emo_logits=softmax(emo_logits)

pos_prob=softmax_emo_logits[0]
neg_prob=softmax_emo_logits[1]

# top_next = [tokenizer.decode(i.item()).strip() for i in probs.topk(topk)[1]]
# top_logits = [probs[i].item() for i in probs.topk(topk)[1]]  # logits for each token
#
# pos_index=top_next.index('positive')
# neg_index=top_next.index('negative')
# softmax_logits = softmax(top_logits)
#
# pos_score=softmax_logits[pos_index]
# neg_score=softmax_logits[neg_index]
# print("positive score is {}".format(pos_score))
# print("negative score is {}".format(neg_score))



