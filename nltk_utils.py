import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()


def tokenize(sentance):
    return nltk.word_tokenize(sentance)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    '''
    sentance = ['hello','how', 'are', 'you']
    words    = ['Hey', 'How', 'are', 'you', 'Is', 'anyone', 'there',]
    bag      = [ 0   ,   0   ,  1   ,   1  ,  0 ,   0   ,  0]

    '''

    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.

    return bag
