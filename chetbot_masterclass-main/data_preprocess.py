import json
import numpy as np
from sklearn.utils import shuffle
from nltk_utils import *

def create_chatbot_vocabulary(file_path):
    # Read the intent file
    with open(file_path,'r') as file:
        intents = json.load(file)

    all_words = []     # Emptpy list to store all the tokined sentences
    tags = []          # Empty list to store the intents tags
    xy = []            # Empyty list to store tokenized words and their corresponding tags as a tuple

    # loop through the intents lists
    for intent in intents['intents']:
        # store the intent tag in the tags list
        tags.append(intent['tag'])
        # loop through the intent patterns
        for pattern in intent['patterns']:
            # apply tokenization on the underlying pattern 
            words = tokenize(pattern)
            # add the tokenized words to the list of words/tokens
            all_words.extend(words)
            # store the tag and the their corresponding tags
            xy.append((tags[-1],words))
    return sorted(all_words),sorted(tags),xy

    
def clean_chatbot_vocab(vocabulary):
    # Remove punctuations from the list of words or tokens
    ignore_list = ['?','!','.',',']
    return sorted(set([stem(word) for word in vocabulary if word not in ignore_list]))

def create_train_data(vocabulary,tags,patterns):
    X_train = []
    y_train = []

    for tag,tokenized_pattern in patterns:
        # Apply bag of word on the underlying tokenized pattern
        bag = bag_of_words(tokenized_pattern=tokenized_pattern,vocabulary=vocabulary)
        # Store the numeric representation of the pettern sample on the train set
        X_train.append(bag)
        # Store the corresponding tag label on the train set
        y_train.append(tags.index(tag))
    return np.array(X_train), np.array(y_train)




# Create the chatbot vocabulary
all_words, tags, tokenized_patterns = create_chatbot_vocabulary(file_path="intent.json")

# Clean the chatbot vocabulary
vocabulary = clean_chatbot_vocab(vocabulary=all_words)

# Create the train set
X_train, y_train = create_train_data(
        vocabulary= vocabulary,
        tags= tags,
        patterns= tokenized_patterns
)

X_train, y_train = shuffle(X_train,y_train,random_state=42)

