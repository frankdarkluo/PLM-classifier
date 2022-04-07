import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from utils import softmax
import numpy as np
tokenizer = RobertaTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = RobertaForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

inputs = tokenizer("it 's small yet they make you feel right at home .", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

logits=softmax(logits.cuda())
predicted_class_id = logits.argmax().item()
outputs={}
outputs['label']=model.config.id2label[predicted_class_id]
outputs['score']=logits.squeeze()[predicted_class_id]