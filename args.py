import argparse


def get_args():

    parser = argparse.ArgumentParser(description="model parameters")
    parser.add_argument('--output_dir', type=str, default="output/", help='Output directory path to store checkpoints.')
    parser.add_argument('--outfile', type=str, default="", help='Output data filepath for predictions on test data.')

    ## Model building

    parser.add_argument('--model_name_or_path', type=str, default="roberta-base",help='model_name_or_path.')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="roberta-base",help='tokenizer_name_or_path.')
    parser.add_argument('--max_len', type=int, default=16,help='Input length of model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--max_key', default=10, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--style_mode', default='plm', type=str,help='plm | pipeline | textcnn')
    parser.add_argument('--class_name',default='EleutherAI/gpt-neo-2.7B',type=str)
    parser.add_argument('--topk', default=5, type=int,help="top-k words in masked out word prediction")
    parser.add_argument("--direction", type=str, default='1-0',help='0-1 | 1-0')
    parser.add_argument("--fluency_weight", type=int, default=1, help='fluency')
    parser.add_argument("--sent_weight",type=int, default=1, help='semantic similarity')
    parser.add_argument("--keyword_weight", type=float, default=8)
    parser.add_argument("--style_weight", type=int, default=1, help='style')
    parser.add_argument("--t_init",type=float,default=3e-2)
    parser.add_argument("--C", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument('--task', default='sentiment', type=str,help='sentiment | formality')

    ## Ablation Study:
    parser.add_argument("--semantic_mode", default='kw-sent',type=str,help='kw | sent | kw-sent')
    parser.add_argument("--action",default='all', type=str, help='replace | delete | insert | all')
    parser.add_argument('--keyword_pos', default=True, type=bool)
    parser.add_argument("--early_stop",default=True, type=bool)

    args, unparsed = parser.parse_known_args()
    return args

