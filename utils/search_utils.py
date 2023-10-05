# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 21:19:58 2023

@author: atr1n17
"""
from utils.preprocessor import preprocess_text
import nltk
import re

def search_documents(keywords, documents):
    """
    Search a list of documents for those that contain one or more of the specified keywords.

    Args:
    keywords (list of str): A list of strings representing the keywords to search for.
    documents (list of dicts): A list of dicts (with name of document and text in document) representing the list of documents to search in.

    Returns:
    A list of dicts representing the names of documents that contain one or more of the specified keywords and the document where keyword is present.
    """
    matching_documents = []
    
    preprocessed_documents= [preprocess_text(doc['text']) for doc in documents]
    tokenized_documents = [nltk.word_tokenize(doc) for doc in preprocessed_documents]
    
    for document, tokens in zip(documents, tokenized_documents):
        token_set = set(tokens)
        if any(keyword in token_set for keyword in keywords):
            matching_documents.append(document)

    return matching_documents 

def get_extracts(keywords, document):
    """
    Search a document for specified keywords.

    Args:
    keywords (list of str): A list of strings representing the keywords to search for.
    documents (dict): Dicts (with name of document and text in document) representing the document to search in.

    Returns:
    A list of dicts with the extracts, pagenum, matched keyword and name of document where keyword is present.
    """
    pages = re.split("<<PAGENUM>>(\d+)<<PAGENUM>>", document['text'])
    if len(pages) > 1:
        pages = pages[1:]
        page_nums = [pages[i] for i in range(len(pages)) if i % 2 == 0]
        pages = [pages[i] for i in range(len(pages)) if i % 2 == 1]
    else:
        page_nums = ['0']
        
    extracts = []
    
    for page_num, page in zip(page_nums, pages):

        sentences = nltk.sent_tokenize(page)
        for sent in sentences:
            preprocessed_sent = preprocess_text(sent)
            tokens = nltk.word_tokenize(preprocessed_sent)
            for keyword in keywords:
                if keyword in tokens:
                    extract = {}
                    extract['page_num'] = page_num
                    extract['extract'] = sent
                    extract['keyword'] = keyword
                    extract['document'] = document['name']
                    extract['doc_rank'] = document['doc_rank'] if 'doc_rank' in document.keys() else 0
                    extracts.append(extract)
    return extracts
