import json
import openai
import gpt
import numpy as np
import pandas as pd
import argparse
from nltk import wordpunct_tokenize
from tqdm import tqdm
word_pairs = {"ca n't": "can not", "n't": "not", "wo n't": "will not","it 's": "it's", "do n't": "don't",
                          "does n't": "doesn't", "did n't": "didn't", "you 'd": "you'd", "you 're": "you're",
                          "you 'll": "you'll", "we 'll": "we'll", "i 'm": "i'm", "I 'd": "I'd", "they 're": "they're",
                          "that 's": "that's", "what 's": "what's", "could n't": "couldn't", "i 've": "i've", "we 've": "we've",
                          "ca n't": "can't", "i 'd": "i'd", "are n't": "aren't", "is n't": "isn't", "was n't": "wasn't",
                          "would n't": "wouldn't", "were n't": "weren't", "wo n't": "won't", "she 's": "she's",
                          "there 's": "there's", "there 're": "there're", "i 'll": "i'll", "he 'd": "he'd", "he 's": "he's"}

def main(args):

    openai.api_key='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'
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
    postfix = "}. Here is a rewrite of the text, which is more negative: {"


    output_path = './output/' +'gpt3-babbage-001_senti_1-0' + '.txt'
    with open(output_path, 'w', encoding='utf8') as of, \
            open(args.data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            sent = line.strip()
            for k, v in word_pairs.items():
                sent = sent.replace(k, v)
            prompt = prefix + str(sent) + postfix
            # input_ids = wordpunct_tokenize(prompt)
            max_len=25
            response=openai.Completion.create(
            engine="text-babbage-001",
            prompt=prompt,
                max_tokens=max_len,
                temperature=0.8)

            content =response.choices[0].text.strip()
            of.write('Generated:' + '\n')
            of.write(str(content)+'\n\n')
            of.flush()


#def gpt3(stext):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/yelp/test.1')
    parser.add_argument("--mode", type=str, default='train')
    args = parser.parse_args()
    print("let's start")
    if args.mode == "train":
        main(args)