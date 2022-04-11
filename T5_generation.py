from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
seq = ' the sentiment of the text { ever since joes has changed hands it is just gotten worse and worse . } is'
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()
model.spread_on_devices()

# inference
input_ids = tokenizer(seq, return_tensors="pt").input_ids.cuda()
# Convert indexed tokens in a PyTorch tensor
# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]
# Get the predicted next sub-word
# if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
probs = predictions[0, -1, :]

pos_logits=probs[tokenizer.encode('positive')]
neg_logits=probs[tokenizer.encode('negative')]
emo_logits=torch.concat([pos_logits,neg_logits])
# softmax_emo_logits=softmax(emo_logits)
#
# pos_prob=softmax_emo_logits[0]
# neg_prob=softmax_emo_logits[1]
