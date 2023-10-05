# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:15:24 2023
@author: atr1n17
"""

import logging
import random
import pandas as pd
import os
import sys
sys.path.insert(0,'./search')
sys.path.insert(0,'./generate')

import search.tfidf_search as s
from data_loader.read_data import read_notes, read_corpus
from generate.lm import load_model, generate_text

# Constants
LOG_FILENAME = 'sleeper.log'
LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
MODE = 'instruct'
MODEL_NAME = 'google/flan-t5-base'
CSV_FILENAME = 'sleeper_memories.csv'
DREAM_FILE = 'sleeper_dreams.txt'
NUM_EXTRACTS_FOR_DREAMS = 5
NUM_KEYWORDS = 5
NUM_DOCS = 1
MAX_DREAM_LENGTH = 50

def load_resources(corpus_folder, notes_folder):
    """Load necessary resources."""
    print("Loading data...")
    corpus = read_corpus(corpus_folder)
    notes = read_notes(notes_folder)
    notes = sum(notes.values(),[])
    print("Loading memories...")
    extracts = s.main(notes, corpus, NUM_KEYWORDS, NUM_DOCS) 
    print("Loading dreams...")
    model, tokenizer = load_model(MODE, MODEL_NAME)
    return extracts, model, tokenizer

def sleeper_memories(extracts):
    """Display sleeper memories."""
    extract = random.choice(extracts)
    print(f"**********************************\nDocument : {extract['document']}\nPage : {extract['page_num']}\nKeyword : {extract['keyword']}\n\n{extract['extract']}\n*********************************")

def sleeper_dreams(extracts, model, tokenizer):
    """Generate and display sleeper dreams."""
    dream_extracts = random.choices(extracts, k=NUM_EXTRACTS_FOR_DREAMS)
    prompt = "Write a story using words from the following text?\n\n Text:"+ ' '.join([extract['extract'] for extract in dream_extracts]) 
    dream = generate_text(model, tokenizer, prompt, MAX_DREAM_LENGTH)
    print(f"*************************************** \n{dream}\n**********************************")
    return dream

def see_memories(extracts):
    """Display extracts from memories."""       
    for extract in extracts:
        print(f"**********************************\nDocument : {extract['document']}\nPage : {extract['page_num']}\nKeyword : {extract['keyword']}\n\n{extract['extract']}\n*********************************")

def save_memories(extracts, dreams, save_path):
    """Save extracts to a CSV file.""" 
    df = pd.DataFrame(extracts)
    df.to_csv(os.path.join(save_path, CSV_FILENAME), sep='\t', encoding='utf-8')
    with open(os.path.join(save_path, DREAM_FILE), 'w') as f:
        f.write('\n\n'.join(dreams))

def main():
    # Configure logging
    logging.basicConfig(filename=LOG_FILENAME, filemode='w', format=LOG_FORMAT, level=LOG_LEVEL)

    # Get folder paths from user
    corpus_folder = input("Enter path to readinglist folder:\n")
    notes_folder = input("Enter path to notes folder:\n")

    # Load resources
    extracts, model, tokenizer = load_resources(corpus_folder, notes_folder)
    dreams = []
    
    while True:
        user_input = input("Choose an option: \n1. Sleeper Memories \n2. Sleeper Dreams\n3.Save Memories\n")
        
        try:
            if user_input.lower() == 'q':
                break
            elif user_input == '1':
                sleeper_memories(extracts) 
            elif user_input == '2':
                dream = sleeper_dreams(extracts, model, tokenizer)
                dreams.append(dream)
                expand = input("See memories? (Y/N)\n")
                if expand.lower() == 'y':
                    see_memories(extracts)
            elif user_input == '3':
                save_path = input("Enter Save Path : \n")
                save_memories(extracts, dreams, save_path)
            else:
                print("Please enter 1 for sleeper insights, 2 for sleeper dreams, or Q to quit")
                pass

        except Exception as e:
            # Log the error
            logging.exception("An error occurred: " + str(e))
            print("An error occurred. Please send the logs to the developer.")

    print("Exiting...")

if __name__ == '__main__':
    main()
