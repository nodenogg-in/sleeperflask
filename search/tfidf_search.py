# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 06:20:07 2023

@author: atr1n17
"""

'''
This script performs TF-IDF-based keyword search on a corpus of documents using a set of notes as the query.

'''
import sys
sys.path.insert(0,'..')
from utils.preprocessor import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
from utils.search_utils import search_documents, get_extracts
from data_loader.read_data import read_corpus, read_notes
import argparse
import pandas as pd
import os


def get_keywords_and_similar_docs(query, corpus, num_keywords, num_docs):
    
    #Preprocess query and documents
    preprocessed_query = preprocess_text(query)
    preprocessed_documents = [preprocess_text(doc['text']) for doc in corpus]
   
    
    tfidf_vectorizer = TfidfVectorizer()
    #Learn vocabulary and idf from the corpus and calculate tf-idf for terms in the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)
    
    #Calculate tf-idf for terms in the query(notes).
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    
    #Get list of terms in the corpus
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    #Sort terms based on tfidf score to get top n keywords
    tfidf_sorting = np.argsort(query_vector.toarray()).flatten()[::-1]
    
    keywords = feature_names[tfidf_sorting][:num_keywords]
    keywords = [preprocess_text(keyword) for keyword in keywords]
    
    #Calculate similarities of tf-idf vectors of the documents in the corpus with tf-idf vector of the query(notes)
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector ).flatten()

    #Sort to get the documents that are most similar to the notes
    sorted_indices = cosine_similarities.argsort()[::-1]
    sorted_documents = [corpus[i] for i in sorted_indices]
    
    most_similar_documents = sorted_documents[:num_docs]
    
    # Add document rank to the document
    for idx, doc in enumerate(most_similar_documents):
        doc['doc_rank'] = idx

    return keywords, most_similar_documents

def main(notes, corpus, num_keywords =5, num_docs=2):
    query = ' '.join(notes)
    keywords, most_similar_documents = get_keywords_and_similar_docs(query, corpus, num_keywords, num_docs)
    print('**********KEYWORDS**********')
    print('\n'.join(keywords))
    extracts = []
    matching_documents = search_documents(keywords, most_similar_documents)
    print('**********DOCUMENTS**********')
    print('\n'.join([document['name'] for document in matching_documents]))
    for document in matching_documents:
        extracts += get_extracts(keywords, document)
    return extracts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', help='path to the corpus folder')
    parser.add_argument('notes_path', help='path to the notes folder')
    parser.add_argument('save_path', help='path to save output')
    parser.add_argument('--num_keywords', default=5, type=int, help='number of keywords to extract')
    parser.add_argument('--num_docs', default=None, type=int, help='number of similar documents to return')

    args = parser.parse_args()

    corpus = read_corpus(args.corpus_path)
    notes = read_notes(args.notes_path)
    
    if args.num_docs is None:
        args.num_docs = len(corpus)
        
    for filename in notes.keys():
        extracts = main(notes[filename], corpus, args.num_keywords, args.num_docs)
        df = pd.DataFrame(extracts)
        df.to_csv(os.path.join(args.save_path, f"{filename[:-5]}.csv"), sep='\t', encoding='utf-8')
        print(f'Extracts are saved in {filename[:-5]}.csv')

