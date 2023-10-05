# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 17:00:47 2023

@author: atr1n17
"""
import numpy as np
import argparse
import sys
sys.path.insert(0,'..')

from gensim.models import Word2Vec
from data_loader.read_data import load_corpus_text


def train_word2vec(sentences, min_count=1):
    """
    Train a Word2Vec model with the given sentences.

    Args:
        sentences: A list of sentences. Each sentence is a list of words.
        min_count: The minimum number of word occurrences for it to be included in the vocabulary.

    Returns:
        The trained Word2Vec model.
    """
    
    model = Word2Vec(vector_size=100, window=5, min_count=min_count, workers=4)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)

    return model


def word_arithmetic(operation, model):
    """
    Perform word arithmetic and find the closest word in the vocabulary.

    Args:
        operation: The word arithmetic operation, e.g., "king - man + woman".
        model: The trained Word2Vec model.

    Returns:
        The word in the vocabulary that is closest to the result of the word arithmetic.
    """
    parts = operation.split()

    # Start with the embedding of the first word
    result_vector = np.array(model.wv[parts[0]])

    # Iterate through the rest of the parts, performing the corresponding operations
    for i in range(1, len(parts), 2):
        if parts[i] == '+':
            result_vector += np.array(model.wv[parts[i + 1]])
        elif parts[i] == '-':
            result_vector -= np.array(model.wv[parts[i + 1]])
        else:
            raise ValueError(f"Invalid operator '{parts[i]}'. Only addition (+) and subtraction (-) are supported.")

    result_vector = result_vector / np.linalg.norm(result_vector)
    similarities = model.wv.cosine_similarities(result_vector, model.wv.vectors)
    top_indices = np.argpartition(-similarities, 5)[:5]
    top_words = [model.wv.index_to_key[index] for index in top_indices]

    return top_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec model and perform word arithmetic.")
    parser.add_argument('--data_path', type=str, help='Path to the directory containing the text files.')
    parser.add_argument('--save_path', type=str,  default=None, help='Path to save the trained model.')
    parser.add_argument('--load_path', type=str,  default=None, help='Path to load the pre-trained model.')
    parser.add_argument('--operation', type=str, help='Word arithmetic operation.')
    args = parser.parse_args()

    sentences = load_corpus_text(args.data_path)

    if args.load_path:
        model = Word2Vec.load(args.load_path)
    else:
        model = train_word2vec(sentences)
        model.save(args.save_path)

    closest_words = ', '.join(word_arithmetic(args.operation, model))
    print(f'Result of "{args.operation}" is "{closest_words}"')
