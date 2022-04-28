perl multi-bleu.perl data/yelp/pos2neg_50references/reference0.1.txt \
data/yelp/pos2neg_50references/reference1.1.txt \
data/yelp/pos2neg_50references/reference2.1.txt \
data/yelp/pos2neg_50references/reference3.1.txt < "$1"

python3 sentiment.py --outfile "$1"
python3 ppl.py --outfile "$1"
