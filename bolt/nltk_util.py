import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemer= PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemer.stem(word.lower())
def bad_of_words(tokenize_sentence, all_words):
    pass


