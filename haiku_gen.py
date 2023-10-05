import sys
import random
import nltk
from nltk.corpus import cmudict

sys.path.insert(0,'./search')
sys.path.insert(0,'./generate')

from search.tfidf_search import main as s_main
from data_loader.read_data import read_notes, read_corpus
from generate.lm import load_model, generate_text

# Download the cmudict if not already downloaded
nltk.download('cmudict')

# Creating dictionary from cmudict
d = cmudict.dict()
NUM_KEYWORDS = 30
NUM_DOCS = 2

def count_syllables(word):
    """
    Function to count syllables in a word using the CMU dictionary.
    If word is not found in dictionary, use syllable_count_for_unknown_word.
    """
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        return syllable_count_oov(word)


def syllable_count_oov(word):
    """
    Function to estimate syllable count for a word not found in CMU dictionary.
    This is a simple rule-based approach that won't be perfect.
    """
    count = 0
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")
    
    if word[0] in vowels:
        count += 1
    if word[-1] == "e":
        count -= 1
    for idx in range(1, len(word)):
        if word[idx] in vowels and word[idx-1] not in vowels:
            count += 1
    if count == 0:
        count += 1
    return count


def get_words_for_first_n_syllables(text, n):
    """
    Function to extract words adding up to a specific number of syllables from the beginning of a text.
    """
    syllables = 0
    words = text.lower().split()
    result = []
    for word in words:
        syllables += count_syllables(word)
        result.append(word)
        if syllables >= n:
            break
    return result


def get_words_for_last_n_syllables(text, n):
    """
    Function to extract words adding up to a specific number of syllables from the end of a text.
    """
    syllables = 0
    words = text.lower().split()
    result = []
    for i in range(len(words)-1, -1, -1):
        word = words[i]
        syllables += count_syllables(word)
        result.append(word)
        if syllables >= n:
            break
    result.reverse()
    return result


def random_split(sentence):
    """
    Function to split a sentence at a random index.
    """
    words = sentence.split()
    split_index = random.randint(1, len(words) - 1)
    return ' '.join(words[:split_index]), ' '.join(words[split_index:])


def generate_line_from_extracts(extracts, num_syllables):
    """
    Function to generate prompt for user input
    """
    extract = random.choice(extracts)
    extract = random_split(extract['extract'])[1]
    words = ' '.join(get_words_for_first_n_syllables(extract, num_syllables)[:num_syllables])
    return words


def generate_line_from_lm(prompt, model, tokenizer, extracts, num_syllables):
    """
    Function to generate text based on a given prompt
    """
    dream = generate_text(model, tokenizer, prompt , num_syllables)
    dream = dream.replace('<pad>', '')
    dream = dream.replace('</s>', '')
    dream = dream.replace('<unk>', '')
    try:
        final_text = ' '.join(get_words_for_last_n_syllables(dream, num_syllables)[:num_syllables])
    except:
        extract = random.choice(extracts)
        extract = random_split(extract['extract'][1:-1])[1]
        final_text = ' '.join(get_words_for_last_n_syllables(extract, num_syllables)[:num_syllables])
    return final_text


def trim(s1, s2):
    """
    Removes any overlap from the beginning of the second string.
    """

    len_s1 = len(s1)
    for i in range(len_s1, 0, -1):
        if s1[-i:] == s2[:i]:
            return s2[i:]
    return s2


def main():
    """
    Main function to load data, models and generate output based on user inputs.
    """
    model, tokenizer = load_model('instruct', 'google/flan-t5-base')
    # Get folder paths from user
    corpus_path = input("Enter path to readinglist folder:\n")
    notes_path = input("Enter path to notes folder:\n")
    num_syllables = int(input("Enter number of syllables:\n"))
    corpus = read_corpus(corpus_path)
    notes = read_notes(notes_path)
    notes = sum(notes.values(),[])

    while True:
        extracts = s_main(notes, corpus, NUM_KEYWORDS, NUM_DOCS)
        prompt = generate_line_from_extracts(extracts, num_syllables)
        user_input = input(prompt+'\n')
        generated_text = generate_line_from_lm("Complete this sentence using 5 syllables:\n\n"+prompt + " " + user_input + " ", model, tokenizer, extracts, num_syllables)
        final_text = trim(prompt, generated_text)
        print(final_text + '.')
        notes = notes + [prompt, user_input, final_text]

        if user_input == 'q':
            break


if __name__ == "__main__":
    main()
