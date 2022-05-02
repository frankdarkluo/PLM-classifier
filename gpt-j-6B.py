#!/usr/bin/env python
# coding: utf-8
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch
import torch.distributed as dist
import argparse
import os
from tqdm import tqdm
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel,GPTJForCausalLM,AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate(model, tokenizer,args):
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    dataset=args.data_path.split('/')[2]
    if args.task=="zero-shot":
        if dataset=='yelp':
            prefix = "Here is some text: {"
            postfix = "}. Rewrite it to be more positive: {"
        else:
            prefix = "Change the formality of the text to be opposite:{ "

    elif args.task == "few-shot":

        if dataset == 'yelp':
            if args.direction=='0-1': label='positive'
            else: label='negative'
            prefix = "Here is some text: {When the doctor asked Linda to take the medicine, he smiled and gave her a lollipop.}. Here \
            is a rewrite of the text, which is more scary. {When the doctor told Linda to take the medicine, there had been \
            a malicious gleam in her eye that Linda didn’t like at all.} Here is some text: {they asked loudly, over the \
            sound of the train.}. Here is a rewrite of the text, which is more intense. {they yelled aggressively, over the \
            clanging of the train.} Here is some text: {When Mohammed left the theatre, it was already dark out}. Here is \
            a rewrite of the text, which is more about the movie itself. {The movie was longer than Mohammed had expected, \
            and despite the excellent ratings he was a bit disappointed when he left the theatre.} Here is some text: {next \
            to the path}. Here is a rewrite of the text, which is about France. {next to la Siene} Here is some text: {The \
            man stood outside the grocery store, ringing the bell.}. Here is a rewrite of the text, which is about clowns. \
            {The man stood outside the circus, holding a bunch of balloons.} Here is some text: {the bell ringing}. Here is \
            a rewrite of the text, which is more flowery. {the peales of the jangling bell} Here is some text: {against the \
            tree}. Here is a rewrite of the text, which is include the word 'snow'. {against the snow-covered bark of the tree} Here is some text: {"
            postfix = "}. Here is a rewrite of the text, which is more "+label+": {"
            word_pairs = {"ca n't": "can not", "n't": "not", "wo n't": "will not","it 's": "it's", "do n't": "don't",
                          "does n't": "doesn't", "did n't": "didn't", "you 'd": "you'd", "you 're": "you're",
                          "you 'll": "you'll", "we 'll": "we'll", "i 'm": "i'm", "I 'd": "I'd", "they 're": "they're",
                          "that 's": "that's", "what 's": "what's", "could n't": "couldn't", "i 've": "i've", "we 've": "we've",
                          "ca n't": "can't", "i 'd": "i'd", "are n't": "aren't", "is n't": "isn't", "was n't": "wasn't",
                          "would n't": "wouldn't", "were n't": "weren't", "wo n't": "won't", "she 's": "she's",
                          "there 's": "there's", "there 're": "there're", "i 'll": "i'll", "he 'd": "he'd", "he 's": "he's"}

    # X = prefix.count('=>')
    # print("it is {}-shot learning".format(X))


    #output_path='./output/'+str(args.model_name.split('/')[1])+'_'+str(dataset)+'.txt'
    output_path='./output/{}_{}_{}.txt'.format(str(args.model_name.split('/')[1]),str(dataset),args.direction)
    with open(output_path,'w',encoding='utf8') as of,\
            open(args.data_path,'r',encoding='utf8') as f:
        lines=f.readlines()
        for line in tqdm(lines):
            torch.cuda.empty_cache()
            sent=line.strip()
            for k, v in word_pairs.items():
                sent = sent.replace(k, v)
            prompt = prefix+ str(sent) +postfix
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

            max_len=len(input_ids[0])+30
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                # bos_token_id=tokenizer.bos_token_id,
                num_return_sequences=1,
                max_length=max_len)

            output_sent=tokenizer.batch_decode(output)[0]
            if args.task == 'few-shot':
                # output_sent=''.join(output_sent.split('\n')[X+1:X+2])
                output_sent = ''.join(output_sent.split(postfix)[-1])
            else:
                output_sent = ''.join(output_sent.split(postfix)[-1])
            of.write('Generated:' + '\n')
            of.write(str(output_sent)+'\n\n')
            of.flush()

def main(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if 'gpt-neo' in args.model_name:
        model= GPTNeoForCausalLM.from_pretrained(
            args.model_name, revision="float16", low_cpu_mem_usage=True)
    elif 'gpt-j' in args.model_name:
        model = GPTJForCausalLM.from_pretrained(
            args.model_name, revision="float16", low_cpu_mem_usage=True)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    generate(model, tokenizer, args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='EleutherAI/gpt-neo-1.3B', help="super large PLM")
    parser.add_argument('--local_rank', type=int, default=-1,help = 'node rank for distributed training')
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--data_path",type=str, default='./data/yelp/test.0')
    parser.add_argument("--task",type=str, default='few-shot')
    parser.add_argument("--direction", type=str, default='1-0')
    args=parser.parse_args()
    print("let's start")
    main(args)


