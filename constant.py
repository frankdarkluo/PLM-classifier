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
stopwords='!"#$%&\'()*+,-–./:;<=>?@[\\]^_`{|}~•…�'+'0123456789'#+'bcdefghjklmnopqrstvwxyz'
postfix = "Sentiment: {"