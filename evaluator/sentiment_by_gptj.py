from transformers import pipeline,GPT2Tokenizer,GPTJForCausalLM,GPTNeoForCausalLM,AutoTokenizer
import argparse
import torch
from utils.functions import softmax
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate(model,tokenizer,args):
    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    output_path='results/'+args.task+'_'+args.direction+'_'+args.model_name.split('/')[-1]+'.txt'
    data_path='./data/yelp/test.'+postfix
    with open(data_path,'r',encoding='utf8') as f, open(output_path,'w',encoding='utf8') as of:
        datas=f.readlines()
        for line in datas:
            line=line.strip()

            input="Sentence: "+str(line)+"\n"
            postfix="Sentiment: {"
            if args.task == 'few-shot':
                prompt = prefix+ input +postfix
            else:
                prompt = input+postfix

            indexed_tokens = tokenizer.encode(prompt)
            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            # Set the model in evaluation mode to deactivate the DropOut modules
            model.eval()

            # Predict all tokens
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            probs = predictions[0, -1, :]

            pos_logits = probs[tokenizer.encode('positive')]
            neg_logits = probs[tokenizer.encode('negative')]

            emo_logits = torch.concat([neg_logits, pos_logits])
            softmax_emo_logits = softmax(emo_logits)

            pos_prob = softmax_emo_logits[1]
            neg_prob = softmax_emo_logits[0]

            label ='negative' if torch.argmax(softmax_emo_logits) ==0 else 'positive'
            # if args.task == 'few-shot':
            #     # output_sent=''.join(output_sent.split('\n')[X+1:X+2])
            #     output_sent = ''.join(output_sent.split('#####')[-1].split('\n\n')[-1])
            # else:
            #     output_sent = ''.join(output_sent.split(postfix)[-1])
            of.write('Generated:' + '\n')
            of.write(input+postfix+label+'}'+'\n\n')
            of.flush()

def main(args):
    model_name=args.model_name
    if 'gpt-neo' in model_name:
        model = GPTNeoForCausalLM.from_pretrained(model_name,revision="float16", low_cpu_mem_usage=True).to(device)
    elif 'gpt-j' in model_name:
        model =GPTJForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    generate(model, tokenizer, args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='EleutherAI/gpt-neo-1.3B', help="super large PLM")
    parser.add_argument("--task",type=str, default='few-shot')
    parser.add_argument("--direction",type=str, default='0-1')
    args=parser.parse_args()
    print("let's start")
    main(args)
