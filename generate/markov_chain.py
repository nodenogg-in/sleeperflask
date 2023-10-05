'''
This script uses Markov Chains to generate text based on a set of input text files.

RUN:
    pip3 install -r requirements.txt
    python markov_chain.py  <corpus_path> <notes_path> <length> [--context_size <context_window>]

Arguments:
    corpus_path (str): Path to the readinglist folder containing the corpus.
    notes_path (str): Path to the notes folder containing the notes.
    length (int): Length of text to be generated.
    --context_size (int): Optional. Size of the context window for the Markov chain. Defaults to 1.

Example:
    python markov_chain.py corpus_folder/ notes_folder/ 100 --context_size 2

'''
import sys
sys.path.insert(0,'..')
import random
from data_loader.read_data import read_corpus, read_notes
import argparse
from collections import defaultdict
from search.tfidf_search import get_keywords_and_similar_docs

def generate_chain(text, order=1):
    '''
    Function to generate a Markov chain from given text.
    
    Args:
        text (str): Input text to generate the Markov chain.
        order (int, optional): Order of the Markov chain. Defaults to 1.
    
    Returns: 
        chain (dict): A dictionary representing the Markov chain.
    '''
    chain = defaultdict(list)
    words = text.split()
    for i in range(len(words) - order):
        key = tuple(words[i:i+order])
        value = words[i+order]
        chain[key].append(value)
    return chain

def generate_text(chain, length=10, seed=None):
    '''
    Function to generate text from a Markov chain.
    
    Args:
        chain (dict): The Markov chain to use for text generation.
        length (int, optional): The length of the text to be generated. Defaults to 10.
        seed (str, optional): Initial seed for the Markov chain. Defaults to None.
    
    Returns: 
        str: A string of generated text.
    '''
    if not seed:
        seed = random.choice(list(chain.keys()))
    else:
        seed = seed.split()
    output = list(seed)
    while(len(output) < length):
        key = tuple(output[-len(seed):])
        if key in chain and chain[key]:
            output.append(random.choice(chain[key]))
        else:
            seed = random.choice(list(chain.keys()))
            output += list(seed)
    return ' '.join(output)

def main(notes, corpus, gen_length, context_window):
    '''
    Main function for generating text with a Markov chain using keywords words from notes.

    Args:
        notes (list of dicts): Notes to extracts keywords from.
        corpus (list of dicts): Corpus to build the Markov chain from.
        gen_length (int): Length of text to be generated.
        context_window (int): The order (context size) of the Markov chain.

    Returns:
        str: A string of generated text.
    '''
    text = ' '.join([doc['text'] for doc in corpus])
    notes = sum(notes.values(),[])
    notes = ' '.join(notes)
    keywords = get_keywords_and_similar_docs(notes, corpus, 1)[0]
    seed = random.choice(keywords)
    print("Chosen seed from notes is : "+ seed)
    chain = generate_chain(text, order=context_window)
    return generate_text(chain, length=gen_length, seed=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', help='path to the readinglist folder')
    parser.add_argument('notes_path', help='path to the notes folder')
    parser.add_argument('length', type=int, help='length of text to be generated')
    parser.add_argument('--context_size', nargs='?', default=1, type=int, help='size of context window for Markov chain')

    args = parser.parse_args()

    # read the input text files
    corpus =  read_corpus(args.corpus_path)
    notes = read_notes(args.notes_path)
    
    
    # generate the Markov chain and the generated text
    generated_text = main(notes, corpus, args.length, args.context_size)
    
    # print the generated text
    print(generated_text)
