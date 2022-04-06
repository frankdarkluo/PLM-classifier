from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering, \
    GPTNeoForCausalLM, GPT2Tokenizer,GPTJForCausalLM
import torch
from torch import nn

topk=1000
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model =GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model =GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
# model=GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",revision = "float16",
#                                       torch_dtype = torch.float16,low_cpu_mem_usage = True).cuda()

seq = 'is the sentiment of the text { ever since joes has changed hands it is just gotten worse and worse . } negative or positive ?'

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

top_next = [tokenizer.decode(i.item()).strip() for i in probs.topk(topk)[1]]
top_logits = [probs[i].item() for i in probs.topk(topk)[1]]  # logits for each token