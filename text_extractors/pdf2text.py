# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:27:50 2023

@author: atr1n17

"""

'''
RUN : 
    pip3 install -r requirements.txt
    python pdf2text.py  <path to pdf or folder containing pdfs> <path to savefile or save folder>
'''


from PyPDF2 import PdfReader
import sys
import codecs
import os


def clean(text):
    text = text.replace('-\n','')
    text = text.replace('\n', ' ')
    return text
    
def extract_pdf(path2pdf):
    """
    Extract text from pdf files

    Args:
    path2pdf (str): A string representing the path to the pdf to be extracted.
   
    Returns:
    String with extracted text
  

    """
    reader = PdfReader(path2pdf, strict=False)
    text = ""
    for pagenum, page in enumerate(reader.pages):
        pagetext = page.extract_text()
        text = text + "<<PAGENUM>>"+str(pagenum)+"<<PAGENUM>>" + pagetext
    return text

if __name__ == '__main__':
    path2pdf = sys.argv[1]
    save_path = sys.argv[2]
    
    if os.path.isfile(path2pdf):
            text = clean(extract_pdf(path2pdf))
            with codecs.open(os.path.join(save_path), 'w', encoding='utf-8', errors='ignore') as f:
                f.write(text)       
    else:    
        for filename in os.listdir(path2pdf):
            if '.pdf' in filename:
                text = clean(extract_pdf(os.path.join(path2pdf, filename)))
                with codecs.open(os.path.join(save_path, filename[:-3]+'txt'), 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(text)

# Common errors seem to be unnecessary hyphenation, missing spaces between words