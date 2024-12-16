import nltk
import numpy as np
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemer= PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemer.stem(word.lower())
def bag_of_words(tokenize_sentence, all_words):
    tokenize_sentence = [stem(w) for w in tokenize_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx]=1.0
    return bag

sentance= ['hello', 'how', 'are', 'you']
words=['hi', 'hello', 'I', 'you', 'bye', 'thank', 'thanks']
bag=bag_of_words(sentance,words)
print(bag)
            


