import nltk
from nltk.corpus import stopwords
#print(stopwords.words('english'))

# a='-)'
# b='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~•'
# a_set = set(a)
# b_set = set(b)
# result = len(a_set & b_set)
# print(result)

from nltk.corpus import wordnet
word_to_test='wa'
if wordnet.synsets(word_to_test):
    print("Yes!!")