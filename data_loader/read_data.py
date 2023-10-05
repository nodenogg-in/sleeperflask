# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:45:19 2023

@author: atr1n17
"""
import json
import codecs
from text_extractors.pdf2text import clean, extract_pdf
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import codecs

def read_corpus(corpus_path):
    """
    Read documents from the specified corpus path and return a list of documents.

    Args:
        corpus_path (str): Path to the folder containing the corpus documents.

    Returns:
        list: A list of dictionaries where each dictionary represents a document with its name and text.
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.pdf'):
            full_filename = os.path.join(corpus_path, filename)
            text = clean(extract_pdf(full_filename))
            with codecs.open(os.path.join(corpus_path, filename[:-4]+'.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
    for filename in os.listdir(corpus_path):
        if filename.endswith('.txt'):
            with codecs.open(os.path.join(corpus_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            continue

        document = {}
        document['name'] = filename
        document['text'] = text
        corpus.append(document)

    return corpus


def read_notes(notes_path):
    """
    Read notes from the specified notes path and return a dict of all notes.

    Args:
        notes_path (str): Path to the folder containing the notes.

    Returns:
        dict: A dict of where filenames are keys and values are strings with the notes.
    """
    all_notes = {}
    for filename in os.listdir(notes_path):
        if '.json' in filename:
            notes = []
            with codecs.open(os.path.join(notes_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                notesjson = json.load(f)
            for node in notesjson['nodes']:
                if 'node_text' in node.keys():
                    notes.append(node['node_text'])
            all_notes[filename] = notes
    return all_notes

def load_corpus_text(corpus_path):
    """
    Load text files from a directory and split them into sentences.

    Args:
        corpus_path: The path of the directory containing the text files.

    Returns:
        A list of sentences, where each sentence is a list of words.
    """
    sentences = []

    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            with codecs.open(os.path.join(corpus_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().lower()

                for sentence in sent_tokenize(text):
                    words = word_tokenize(sentence)
                    sentences.append(words)

    return sentences