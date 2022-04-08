import random
import numpy as np
import torch
from nltk.corpus import stopwords
from args import get_args
args=get_args()
stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

positive_word_lst=['positive','good','better','well']
negative_word_lst=['negative','bad','worse','depressed']
formal_word_lst=['formal']
informal_word_lst=['informal']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x):
    x = x - torch.max(x)
    exp_x = torch.exp(x)
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

def predict_next_word(model,tokenizer,input_text,k,direction):
    indexed_tokens = tokenizer.encode(input_text)
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to(device)
    model.to(device)
    # Predict all tokens
    with torch.no_grad():
      outputs = model(tokens_tensor)
      predictions = outputs[0]

    # Get the predicted next sub-word
    # if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
    probs = predictions[0, -1, :]

    if args.task=='sentiment':
        pos_logits = probs[tokenizer.encode('positive')]
        neg_logits = probs[tokenizer.encode('negative')]
    else:
        pos_logits = probs[tokenizer.encode('formal')]
        neg_logits = probs[tokenizer.encode('informal')]
    emo_logits = torch.concat([pos_logits, neg_logits])
    softmax_emo_logits = softmax(emo_logits)

    pos_prob = softmax_emo_logits[0]
    neg_prob = softmax_emo_logits[1]
    if direction=='0-1':
        output_prob = (1 - neg_prob) * pos_prob  # make the prob more robust
    else: #1-0
        output_prob= (1-pos_prob)*neg_prob

    return output_prob

def pipe(res_cand,direction):
    label = res_cand[0]['label'].lower()
    score = res_cand[0]['score']
    if direction=='0-1':
        classifi_prob = score if label == 'positive' else 1- score
    else: # 1-0
        classifi_prob= score if label == 'negative' else 1-score

    return classifi_prob,label

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
