# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:51:01 2023

@author: atr1n17

pip install -r requirements.txt
python text2text.py <prompt_file> <length of text to generate> <modelname>
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import codecs
import sys
def generate(
    model,
    tokenizer,
    prompt,
    stop_token,
    length,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.0
):

    assert length > 0

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")


    outputs = model.generate(
            input_ids=prompt_tokens,
            max_length=length + len(prompt_tokens[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )

    if len(outputs.shape) > 2:
        outputs.squeeze_()

    output = outputs[0].tolist()
    generated_text = tokenizer.decode(output, clean_up_tokenization_spaces=True)
    return generated_text

model_name = sys.argv[3] if len(sys.argv) >3 else 'google/flan-t5-base'
temperature = 1.0
k=400
p=0.9
repetition_penalty = 3.0
length = int(sys.argv[2])
stop_token = '|endoftext|'
prompt_file = sys.argv[1]
with codecs.open(prompt_file, 'r', encoding = 'utf-8', errors='ignore') as f:
    prompt_text = f.read()
#prompt_text = "Write a plausible story that ends with this paragraph?\n\nLast paragraph:"+ prompt_text 
#prompt_text = "Summarize:\n\n" + prompt_text
#prompt_text = "Write a story to illustrate the following: \n\n"+ prompt_text
#prompt_text = "List game ideas inspired by the following : \n\n" + prompt_text
#prompt_text = "Write a story using words from the following text?\n\n Text:"+ prompt_text 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = generate(
    model,
    tokenizer,
    prompt_text,
    stop_token,
    length,
    temperature = temperature,
    top_k=k,
    top_p=p,
    repetition_penalty = repetition_penalty
)
print("======= Generated Text =========")
print(text)
