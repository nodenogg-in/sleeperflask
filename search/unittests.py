# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:03:35 2023

@author: atr1n17
"""


import unittest
from tfidf_search import get_keywords_and_similar_docs
from utils.search_utils import get_extracts, search_documents

class TestGetExtracts(unittest.TestCase):

    def test_matching_extract(self):
        keywords = ['apple']
        document = {'name': 'doc1', 'text': '<<PAGENUM>>0<<PAGENUM>>I like to eat apples. They are my favorite fruit.<<PAGENUM>>1<<PAGENUM>>', 'doc_rank': 1}
        expected = [{'page_num': '0', 'extract': 'I like to eat apples.', 'keyword': 'apple', 'document': 'doc1', 'doc_rank' : 1}]
        result = get_extracts(keywords, document)
        self.assertEqual(result, expected)

    def test_multiple_keywords(self):
        keywords = ['apple', 'banana']
        document = {'name': 'doc1', 'text': '<<PAGENUM>>0<<PAGENUM>>I like to eat apples and bananas. They are my favorite fruits.<<PAGENUM>>1<<PAGENUM>>', 'doc_rank' : 1}
        expected = [{'page_num': '0', 'extract': 'I like to eat apples and bananas.', 'keyword': 'apple', 'document': 'doc1', 'doc_rank' : 1},                    {'page_num': '0', 'extract': 'I like to eat apples and bananas.', 'keyword': 'banana', 'document': 'doc1', 'doc_rank' : 1}]
        result = get_extracts(keywords, document)
        self.assertEqual(result, expected)

    def test_no_matching_extract(self):
        keywords = ['orange']
        document = {'name': 'doc1', 'text': '<<PAGENUM>>0<<PAGENUM>>I like to eat apples. They are my favorite fruit.<<PAGENUM>>1<<PAGENUM>>'}
        expected = []
        result = get_extracts(keywords, document)
        self.assertEqual(result, expected)

    def test_multiple_pages(self):
        keywords = ['apple']
        document = {'name': 'doc1', 'text': '<<PAGENUM>>0<<PAGENUM>>I like to eat apples. They are my favorite fruit.<<PAGENUM>>1<<PAGENUM>>I also like apple pie.<<PAGENUM>>2<<PAGENUM>>', 'doc_rank' : 1}
        expected = [{'page_num': '0', 'extract': 'I like to eat apples.', 'keyword': 'apple', 'document': 'doc1', 'doc_rank' : 1}, {'page_num': '1', 'extract': 'I also like apple pie.', 'keyword': 'apple', 'document': 'doc1', 'doc_rank' : 1}]
        result = get_extracts(keywords, document)
        self.assertEqual(result, expected)


class TestSearchDocuments(unittest.TestCase):
    
    def setUp(self):
        self.documents = [
            {"name": "doc1.txt", "text": "This is the first document."},
            {"name": "doc2.txt", "text": "This is the second document."},
            {"name": "doc3.txt", "text": "This is the third document."}
        ]
    
    def test_search_documents_one_keyword(self):
        keywords = ["first"]
        expected_matching_documents = [{"name": "doc1.txt", "text": "This is the first document."}]
        matching_documents = search_documents(keywords, self.documents)
        self.assertEqual(matching_documents, expected_matching_documents)
        
    def test_search_documents_multiple_keywords(self):
        keywords = ["second", "third"]
        expected_matching_documents = [
            {"name": "doc2.txt", "text": "This is the second document."},
            {"name": "doc3.txt", "text": "This is the third document."}
        ]
        matching_documents = search_documents(keywords, self.documents)
        self.assertEqual(matching_documents, expected_matching_documents)
        
    def test_search_documents_no_match(self):
        keywords = ["fourth"]
        expected_matching_documents = []
        matching_documents = search_documents(keywords, self.documents)
        self.assertEqual(matching_documents, expected_matching_documents)
        
class TestGetKeywordsAndSimilarDocs(unittest.TestCase):
    
    def test_get_keywords_and_similar_docs(self):
        query = "The quick brown fox jumps over the lazy dog."
        corpus = [
            {"name": "doc1", "text": "The brown dog is quick."},
            {"name": "doc2", "text": "The quick fox is brown."},
            {"name": "doc3", "text": "The lazy cat is not brown."},
            {"name": "doc4", "text": "The dog and the cat are friends."},
            {"name": "doc5", "text": "The quick brown fox jumps over the lazy dog and the lazy cat."}
        ]
        num_keywords = 3
        num_docs = 2
        
        keywords, similar_docs = get_keywords_and_similar_docs(query, corpus, num_keywords, num_docs)
        
        # Test the number of keywords and similar documents returned
        self.assertEqual(len(keywords), num_keywords)
        self.assertEqual(len(similar_docs), num_docs)
        
        # Test the content of the keywords
        self.assertIn("jump", keywords)
        self.assertIn("lazy", keywords)
        
        # Test the content of the similar documents
        self.assertIn(corpus[4], similar_docs)
        self.assertIn(corpus[1], similar_docs)
        self.assertNotIn(corpus[2], similar_docs)
    

if __name__ == '__main__':
    unittest.main()
