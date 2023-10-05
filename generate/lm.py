"""
This script uses a pre-trained language model to generate text based on extracts from the readinglist.

RUN:
    pip install -r requirements.txt
    python lm.py <corpus_path> <notes_path> --length <length_of_text_to_generate> --model_name <model_name>
   
Arguments:
    corpus_path : Path to readinglist.
    notes_path : Path to notes.
    length_of_text_to_generate (int): The desired length of the generated text.
    model_name (str): The name or path of the pre-trained language model to use.

Example:
    python lm.py ./corpus_path ./notes_path --length 100 --model-name google/flan-t5-base

"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import argparse
import torch
import sys
sys.path.insert(0,'..')
import search.tfidf_search as s
from data_loader.read_data import read_corpus, read_notes

# Default values
DEFAULT_AUTOREGRESSIVE_MODEL = 'distilgpt2'
DEFAULT_SEQ2SEQ_MODEL = 'google/flan-t5-base'

def load_model(mode, model_name):
    """
    Load the pre-trained language model and tokenizer based on the mode of operation.

    Args:
        mode (str): The mode of operation. 'continue' for autoregressive language models or 'instruct' for sequence-to-sequence language models.
        model_name (str): The name or path of the pre-trained language model.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    if mode == 'continue':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, length):
    """
    Generate text based on the given prompt using the loaded language model.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the language model.
        prompt (str): The text prompt to generate text from.
        length (int): The desired length of the generated text.

    Returns:
        str: The generated text.
    """
    assert length > 0

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # Create attention mask
    attention_mask = torch.ones_like(prompt_tokens)
    outputs = model.generate(
        input_ids=prompt_tokens,
        attention_mask = attention_mask,
        max_length=length + len(prompt_tokens[0]),
        temperature=1.0,
        top_k=40,
        top_p=0.9,
        repetition_penalty=5.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    if len(outputs.shape) > 2:
        outputs.squeeze_()

    output = outputs[0].tolist()
    generated_text = tokenizer.decode(output, clean_up_tokenization_spaces=True)
    return generated_text


def main(notes, corpus, length = 100):
    '''
    Main function for generating text with a LM using keywords from notes and extracts from the readinglist.

    Arguments:
    notes (list of dicts): Notes to extracts keywords from.
    corpus (list of dicts): Corpus to build th emarkov chain from.
    length (int): Length of text to generate.

    Returns:
    A string of generated text.
    '''
    extracts = s.main(notes, corpus)
    model, tokenizer = load_model('instruct', DEFAULT_SEQ2SEQ_MODEL)
    prompt = "Write a story using words from the following text?\n\n Text:"+ ' '.join([extract['extract'] for extract in extracts]) 
    text = generate_text(model, tokenizer, prompt, length)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained language model.")
    parser.add_argument('corpus_path', help='The path of the directory containing the readinglist.')
    parser.add_argument('notes_path', help='The path of the directory containing the notes.')
    parser.add_argument("--length", type=int, default=100, help="The desired length of the generated text.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_SEQ2SEQ_MODEL, help="The name or path of the pre-trained language model.")
    args = parser.parse_args()
    
    corpus =  read_corpus(args.corpus_path)
    notes = read_notes(args.notes_path)
    
    text = main(notes, corpus, args.length)
    print("======= Generated Text =========")
    print(text)
