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
elif opt.task=='formality':
    prefix = """
    Sentence: i do not intend to be mean ..
    Sentiment: {formal}

    #####

    Sentence: i do n't want to be mean ..
    Sentiment: {informal}

    #####

    Sentence: This movie was actually pretty funny.
    Sentiment: {positive}

    #####

    """
    postfix="Formality: {"
stopwords='!"#$%&\'()*+,-–./:;<=>?@[\\]^_`{|}~•…�'+'0123456789'#+'bcdefghjklmnopqrstvwxyz'
