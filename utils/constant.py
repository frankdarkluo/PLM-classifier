from args import get_args
opt=get_args()
if opt.task=='sentiment':
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
    postfix = "Sentiment: {"
    stopwords='#$%&()*+,-–./:;<=>@[\\]^_`{|}~•…�'+'0123456789'#+'bcdefghjklmnopqrstvwxyz'
elif opt.task=='formality':
    prefix = """
    Sentence: i do not intend to be mean
    Formality: {formal}

    #####

    Sentence: ohhh i don't intend to be mean ..
    Formality: {informal}

    #####

    Sentence: what 're u doing here ?
    Formality: {informal}

    #####
    
    Sentence: people are having coffee , lunch, playing in the park , playing and talking .
    Formality: {formal}

    #####

    """
    postfix="Formality: {"
    stopwords = '#$%|~…�'
