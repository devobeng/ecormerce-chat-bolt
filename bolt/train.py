import json
import numpy as np
from nltk_util import tokenize,stem,bag_of_words
with open('intents.json') as file:
    intents = json.load(file)
    
all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)    
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
print(all_words)
all_words=sorted(set(all_words))
tags=sorted(set(tags))

x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    
    label=tags.index(tag)
    y_train.append(label)
x_train=np.array(x_train)
y_train=np.array(y_train)
