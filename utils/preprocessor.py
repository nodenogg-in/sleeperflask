# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 07:54:32 2023

@author: atr1n17
"""

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
# Download the WordNet lemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
import re
stopwords = set(stopwords.words('english'))


def lemmatize(word, pos_tag):
    """
    Lemmatize a word based on its part of speech tag.

    Args:
    word (str): A string representing the word to lemmatize.
    pos_tag (str): A string representing the word's part of speech tag.

    Returns:
    A string representing the lemmatized word.
    """

    wordnet_tag = {'NN': wordnet.NOUN,
                   'NNS' :wordnet.NOUN,
                   'VBP': wordnet.VERB,                
                   'VBZ': wordnet.VERB,
                   'RB': wordnet.ADV,
                   'RBR': wordnet.ADV,
                   'JJ': wordnet.ADJ}

    tag = wordnet_tag.get(pos_tag[0].upper(), wordnet.NOUN)
 
    lemmatizer = WordNetLemmatizer()


    return lemmatizer.lemmatize(word, tag)

def preprocess_text(text):
    """
    Preprocess text by tokenizing, lemmatizing, and rejoining.

    Args:
    text (str): A string representing the text to preprocess.

    Returns:
    A string representing the preprocessed text.
    """

    text = text.lower()
    
    # Remove html tags
    text = re.sub(r'<.*?>', '', text)

    # Remove all non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    
    
    
    

    tokens = nltk.word_tokenize(text)
    
    tokens = [token for token in tokens if token not in stopwords]

    pos_tags = nltk.pos_tag(tokens)

    lemmatized_tokens = [lemmatize(token, pos_tag) for token, pos_tag in pos_tags]


    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text