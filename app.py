# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:22:42 2023

@author: atr1n17
"""


from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from data_loader.read_data import read_corpus, read_notes, load_corpus_text
from search.tfidf_search import main as tf_main
from search.keyword_search import main as kw_main
from generate.lm import load_model, generate_text, DEFAULT_SEQ2SEQ_MODEL
from generate.markov_chain import generate_text as mk_generate_text, generate_chain
from generate.word_arithmetic import train_word2vec, word_arithmetic
from haiku_gen import generate_line_from_extracts

import json
import random
import os
import zipfile

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
UPLOAD_FOLDER = "data"
NOTES_FOLDER = "notes"
READINGLIST_FOLDER = "corpus"
results = {}
model = None
tokenizer = None
w2v_model = None
    
def set_up():
    global w2v_model, results, model, tokenizer   
    corpus_path = config['corpus_path']
    notes_path = config['notes_path']
    num_keywords = int(config['num_keywords'])
    num_docs = int(config['num_docs'])
    cache_path = config['cache_path']
    corpus = read_corpus(corpus_path)
    notes = read_notes(notes_path)

    if num_docs is None:
        num_docs = len(corpus)
    results = {}
    for group in notes.keys():
        if config['search_model'] == 'kw':
            result = kw_main(notes[group], corpus, num_keywords, num_docs)
        else:
            result = tf_main(notes[group], corpus, num_keywords, num_docs)
        results[group] = result
        
    with open(os.path.join(cache_path, 'all_results.json'), 'w') as f:
        json.dump(results, f)
    if config['gen_model'] =='lm':
        model, tokenizer = load_model('instruct', DEFAULT_SEQ2SEQ_MODEL)
    
    sentences = load_corpus_text(corpus_path)
    # w2v_model = train_word2vec(sentences)
    # w2v_model = None
    return results, model, tokenizer, w2v_model

@app.route("/")
def hello():
    return "Hello Flask!"

@app.route('/status', methods=['POST'])
def status_endpoint(): 
    response = {}
    if w2v_model:
        status = "Awake"

    else:
        status = "Asleep"
    response['sleeper_status'] = status
    return jsonify(response) 

@app.route('/topics', methods=['POST'])
def topics_endpoint(): 
    data = request.get_json()
    group_name = data['group_name']
    result = results[group_name+'.json']
    keywords = list(set([entry['keyword'] for entry in result]))
    response = {}
    response['topics'] = keywords
    sorted_results = sorted(result, key=lambda x: x['doc_rank'])
    docs = list(set([entry['document'] for entry in sorted_results]))
    response['tomes'] = docs
    return jsonify(response) 
 
@app.route('/memories', methods=['POST'])
def memories_endpoint(): 
    data = request.get_json()
    group_name = data['group_name']
    result = random.choice(results[group_name+'.json'])  
    print(result)
    return jsonify(result)       

@app.route('/dreams', methods=['POST'])
def dreams_endpoint(): 
    data = request.get_json()
    group_name = data['group_name']
    result = results[group_name+'.json']
    

    if config['gen_model']=='lm':
        sample_size = min(len(result),5)
        result = random.sample(result, sample_size)
        prompt = "Write an imaginative story using words from the following text?\n\n Text:"+ ' '.join([extract['extract'] for extract in result]) 
        text = generate_text(model, tokenizer, prompt, 100) 
        text = text.replace('<pad>','')
        text = text.replace('</s>','')
    else:
        sample_size = min(len(result),30)
        result = random.sample(result, sample_size)
        text = ' '.join([extract['extract'] for extract in result])
        chain = generate_chain(text, order=2)
        seed = ' '.join(list(random.choice(list(chain.keys()))))
        text = mk_generate_text(chain, length=100, seed=seed)
    return jsonify(text)  

@app.route('/alchemy', methods=['POST'])
def alchemy_endpoint(): 
    data = request.get_json()
    op = data['alchemy_word']
    response = word_arithmetic(op, w2v_model)
    return jsonify(response)        

@app.route('/haiku', methods=['POST'])
def haiku_endpoint(): 
    data = request.get_json()
    group_name = data['group_name']
    result = results[group_name+'.json']
    response = generate_line_from_extracts(result, 5)
    return jsonify(response) 

@app.route('/haikucomplete', methods=['POST'])
def haiku_complete_endpoint(): 
    data = request.get_json()
    group_name = data['group_name']
    #first_line = data['haiku_first_line']
    #second_line = data['haiku_second_line']
    result = results[group_name+'.json']
    response = generate_line_from_extracts(result, 5)
    return jsonify(response) 

@app.route('/upload', methods=['POST'])
def fileUpload():
    
    save_path = request.form['save_path']
    if save_path:
       config["cache_path"] = os.path.join(UPLOAD_FOLDER, save_path)
       config["corpus_path"] = os.path.join(UPLOAD_FOLDER, save_path, READINGLIST_FOLDER)
       config["notes_path"] = os.path.join(UPLOAD_FOLDER, save_path,  NOTES_FOLDER)
    else:
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        # Check if the file is a zip file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
        config["corpus_path"] = os.path.join(UPLOAD_FOLDER, file.filename[:-4], READINGLIST_FOLDER)
        config["notes_path"] = os.path.join(UPLOAD_FOLDER, file.filename[:-4], NOTES_FOLDER)
        config["cache_path"] = os.path.join(UPLOAD_FOLDER, file.filename[:-4])
            
    config['num_keywords'] = request.form['num_keywords']
    config['num_docs'] = request.form['num_docs']
    config['search_model'] = request.form['search_model']
    config['gen_model'] = request.form['gen_model']

    results, model, tokenizer, w2v_model = set_up()
    return 'Set up successful'

@app.route('/end', methods=['POST'])
def endSession():
    global results, model, tokenizer, w2v_model 
    results = {}
    model = None
    tokenizer = None
    w2v_model = None
    return 'Session Ended'

if __name__ == '__main__':  
    config={}
    config['corpus_path']='./'
    config['notes_path']='./'
    config['cache_path']='./'
    config['search_model'] = 'tfidf'
    config['gen_model'] = 'lm'
    config['num_keywords'] = 5
    config['num_docs'] = 2
    
    app.run(host='0.0.0.0', port=5000)