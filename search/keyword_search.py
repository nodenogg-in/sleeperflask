'''
This script performs keyword search on a corpus using notes.

RUN:
    pip3 install -r requirements.txt
    python keyword_search.py  <corpus_path> <notes_path> [--num_keywords <num_keywords>]

Arguments:
    corpus_path (str): Path to the folder containing the corpus documents.
    notes_path (str): Path to the file containing the notes in JSON format.
    --num_keywords (int): Optional. Number of keywords to extract from the notes. Default is 5.

Example:
    python keyword_search.py corpus_folder/ notes_folder/ --num_keywords 10

'''
import sys
sys.path.insert(0,'..')
import nltk
nltk.download('punkt')
from collections import Counter
from utils.preprocessor import preprocess_text
from data_loader.read_data import read_corpus, read_notes
from utils.search_utils import search_documents, get_extracts
import argparse

def extract_top_keywords(text, n):
    """
    Extract the n most common keywords from a given text.

    Args:
    text (str): A string representing the text to extract keywords from.
    n (int): An integer representing the number of top keywords to extract.

    Returns:
    A list of tuples representing the n most common keywords in the text,
    sorted in descending order by their frequency. Each tuple contains a
    keyword and its frequency in the text.
    """
    
    preprocessed_text= preprocess_text(text)
    tokens = nltk.word_tokenize(preprocessed_text)
    counts = Counter(tokens)
    return counts.most_common(n)

    


def main(notes, corpus, num_keywords, num_docs=1):
    """
    Main function for performing keyword search on a corpus using notes.

    Args:
        corpus_path (str): Path to the folder containing the corpus documents.
        notes_path (str): Path to the folder containing the notes.
        num_keywords (int) : Number of keywords to extract from notes.
        num_docs (int): Max number of matching documents to search

    Returns:
        None
    """
    
    notes = ' '.join(notes)
    keywords = extract_top_keywords(notes, num_keywords)
    keywords = [k[0] for k in keywords]
    matching_documents = search_documents(keywords, corpus)
    matching_documents = matching_documents[:num_docs]
    extracts = []
    for document in matching_documents:
        extracts += get_extracts(keywords, document)
    return extracts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform keyword search on a corpus using notes')
    parser.add_argument('corpus_path', type=str, help='Path to the folder containing the corpus documents')
    parser.add_argument('notes_path', type=str, help='Path to the folder containing the notes')
    parser.add_argument('--num_keywords', default=5, type=int, help='number of keywords to extract')
    parser.add_argument('--num_docs', default=1, type=int, help='number of documents to use')
    args = parser.parse_args()
    
    corpus = read_corpus(args.corpus_path)
    notes = read_notes(args.notes_path)
    
    for filename in notes.keys():
        keywords, docs = main(notes[filename], corpus, args.num_keywords, args.num_docs)
        print(filename)
        print('**********KEYWORDS**********')
        print('\n'.join(keywords))
        print('**********DOCUMENTS*********')
        print('\n'.join([doc['name'] for doc in docs]))
        

