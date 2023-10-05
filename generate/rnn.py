'''
RUN:
   pip install requirements.txt
   python rnn.py  <readinglist_path> <notes_path> <save_model_path> --diversity <diversity>
'''
import numpy as np
from keras.layers import LSTM, Dense, Activation, Embedding
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import sys
sys.path.insert(0,'..')
import os
import pickle
import argparse
from utils.preprocessor import preprocess_text
from search.tfidf_search import get_keywords_and_similar_docs
from data_loader.read_data import read_corpus, read_notes
import random
max_len = 25
step = 5
gen_len = 30
batch_size = 128

def create_vocab(text):
    """
    Create a vocabulary from the text.

    Args:
        text (str): The text to process.

    Returns:
        tuple: Two dictionaries, the first mapping words to integers and the second mapping integers to words.
    """
    # split the text into words
    words = text.split()
    # create a mapping from words to integers
    unique_words = sorted(list(set(words)))
    word_to_int = dict((w, i) for i, w in enumerate(unique_words))
    return unique_words, word_to_int

def prepare_data(text, word_to_int, preprocess = False):
    """
    Prepare the training and target data.

    Args:
        text (str): The text to process.
        word_to_int (dict): A dictionary mapping words to integers.
        preprocess (bool, optional): Indicates if the text should be preprocessed. Default is False.

    Returns:
        tuple: Two numpy arrays, the first containing the training data and the second containing the target data.
    """
    if preprocess:
        text = preprocess_text(text)
    words = text.split()
    sentences = []
    next_words = []
    for i in range(0, len(words) - max_len, step):
        sentences.append(words[i:i + max_len])
        next_words.append(words[i + max_len])
    
    X = np.zeros((len(sentences), max_len), dtype=np.int)
    y = np.zeros((len(sentences),), dtype=np.int)
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if word in word_to_int:
                X[i, j] = word_to_int[word]
        if next_words[i] in word_to_int:
            y[i] = word_to_int[next_words[i]]
    
    return X, y

def build_model(input_dim, max_len):
    """
    Build the LSTM model.

    Args:
        input_dim (int): The size of the vocabulary.
        max_len (int): The length of the input sequences.

    Returns:
        Sequential: The constructed LSTM model.
    """
    model = Sequential()
    model.add(Embedding(input_dim, 50, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model  
    
def generate_text(model, seed_sentence, diversity, word_to_int, vocab):
    """
    Generate text using the LSTM model.

    Args:
        model (Sequential): The LSTM model.
        seed_sentence (str): The sentence to start the generation with.
        diversity (float): The diversity of the generated text.
        word_to_int (dict): A dictionary mapping words to integers.
        vocab (list): The vocabulary mapping integers to words.
    
    Returns:
        str: String of generated text.
    """
    generated = []
    sentence = seed_sentence.split()
    generated.extend(sentence)
    sys.stdout.write(' '.join(generated))
    
    for _ in range(gen_len):
        x_pred = np.zeros((1, max_len))
        for j, word in enumerate(sentence):
            if word in word_to_int:
                x_pred[0, j] = word_to_int[word]
        
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(probas)
        next_word = vocab[next_index]
        generated.append(next_word)
        sentence = sentence[1:] + [next_word]
    print(' '.join(generated))

def train(data_path, save_path, epochs, preprocess = False):
    """
    Train an LSTM model.

    Args:
        data_path (str): The path of the directory containing the text files.
        save_path (str): The path to save the trained model.
        epochs (int): The number of epochs to train the model.
        preprocess (bool, optional): Indicates if the text should be preprocessed. Default is False.
        
    Returns:
        tuple: The trained LSTM model, word_to_int dict and vocab dict
    """
    text = ''
    for doc in corpus:
        text = text + '\n' + doc['text'].lower()
    lines = text.split('.')
    min_line_length = 5
    clean_lines = [line for line in lines if len(line.split()) > min_line_length]
    cleaned_text = '. '.join(clean_lines)
    vocab, word_to_int = create_vocab(cleaned_text)
    X, y  = prepare_data(cleaned_text, word_to_int, preprocess = preprocess)
    input_dim = len(vocab)
    model = build_model(input_dim, max_len)
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    return model, word_to_int, vocab


def load_resources(load_path):
    """
    Load LSTM model and resources

    Args:
        load_path (str): The path of the directory containing the trained model and related files.
        
    Returns:
        tuple: The loaded LSTM model, word_to_int dict and vocab dict
    """
    model = load_model(os.path.join(load_path, 'model.h5'))
    with open(os.path.join(load_path, 'word_to_int.pkl'), 'rb') as f:
        word_to_int = pickle.load(f)
    with open(os.path.join(load_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return model, word_to_int, vocab

def save_resources(save_path, model, word_to_int, vocab):
    """
    Save LSTM model and resources

    Args:
        save_path (str): The path of the directory to save the trained model and related files.
        model (Sequential): The LSTM model.
        word_to_int (dict): A dictionary mapping words to integers.
        vocab (list): The vocabulary mapping integers to words.
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    model.save(os.path.join(save_path, 'model.h5'))
    with open(os.path.join(save_path, 'word_to_int.pkl'), 'wb') as f:
        pickle.dump(word_to_int,f)
    with open(os.path.join(save_path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab,f)
    
   

def main(notes, corpus, model_path, diversity = 1.2):
    '''
    Main function for generating text with an LSTM trained on the readinglist using keywords words from notes.

    Arguments:
    notes (list of dicts): Notes to extract keywords from.
    corpus (list of dicts): Corpus to build the Markov chain from.
    model_path (str): Path to load model from or save model to.
    diversity (float, optional): Diversity of text to be generated, default is 1.2

    Returns:
    str: A string of generated text.
    '''
    text = ' '.join([doc['text'] for doc in corpus])
    notes = sum(notes.values(),[])
    notes = ' '.join(notes)
    keywords = get_keywords_and_similar_docs(notes, corpus)[0]
    seed = random.choice(keywords)
    print("Chosen seed from notes is : "+ seed)
    if 'model.h5' in os.listdir(model_path):
        model, word_to_int, vocab = load_resources(model_path) 
    else:
        epochs = 5
        model, word_to_int, vocab = train(corpus, model_path, epochs)
        save_resources(model_path, model, word_to_int, vocab)

        
    text =  generate_text(model, seed, diversity, word_to_int, vocab)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model or generate text with a trained model.")

    parser.add_argument('corpus_path', help='The path of the directory containing the readinglist.')
    parser.add_argument('notes_path', help='The path of the directory containing the notes.')
    parser.add_argument('model_path', help='The path to save/load the trained model.')
    parser.add_argument('--diversity', type=float, nargs='?', default=1.2, help='The diversity of the generated text.')

    args = parser.parse_args()
    # read the input text files
    corpus =  read_corpus(args.corpus_path)
    notes = read_notes(args.notes_path)

        
    main(notes, corpus, args.model_path, args.diversity)



   

