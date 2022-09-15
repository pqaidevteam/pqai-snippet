import re
import json
from pathlib import Path
import numpy as np

BASE_DIR = str(Path(__file__).parent.parent.resolve())
models_dir = "{}/assets".format(BASE_DIR)

"""
Load stopwords
"""
stopword_file = models_dir.rstrip("/") + "/stopwords.txt"
with open(stopword_file, "r") as file:
    stopword_list = file.read().strip().splitlines()
stopword_dict = {word: 1 for word in stopword_list}


def is_generic(word):
    """Check if a given word is a generic word, e.g., 'the', 'of', etc.
    It is determined on the basis of a hand-picked list of keywords
    determined as generic words commonly used in patents.
    
    Args:
        word (str): Word to be checked.
    
    Returns:
        bool: True if the word is a generic word, False otherwise.
    """
    return True if word in stopword_dict else False


def get_sentences(text):
    """Split a given (English) text (possibly multiline) into sentences.
    
    Args:
        text (str): Text to be split into sentences.
    
    Returns:
        list: Sentences.
    """
    sentences = []
    paragraphs = get_paragraphs(text)
    ends = r"\b(etc|viz|fig|FIG|Fig|e\.g|i\.e|Nos|Vol|Jan|Feb|Mar|Apr|\
    Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ser|Pat|no|No|Mr|pg|Pg|figs|FIGS|Figs)$"
    for paragraph in paragraphs:
        chunks = re.split(r"\.\s+", paragraph)
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if re.search(ends, chunk) and i < len(chunks) - 1:
                chunks[i] = chunk + ". " + chunks[i + 1]
                chunks.pop(i + 1)
            elif i < len(chunks) - 1:
                chunks[i] = chunks[i] + "."
            i += 1
        for sentence in chunks:
            sentences.append(sentence)
    return sentences


def get_paragraphs(text):
    r"""Split a text into paragraphs. Assumes paragraphs are separated
    by new line characters (\n).
    
    Args:
        text (str): Text to be split into paragraphs.
    
    Returns:
        list: Paragraphs.
    """
    return [s.strip() for s in re.split("\n+", text) if s.strip()]


def tokenize(text, lowercase=True, alphanums=False):
    """Get tokens (words) from given text.
    
    Args:
        text (str): Text to be tokenized (expects English text).
        lowercase (bool, optional): Whether the text should be
            lowercased before tokenization.
        alphanums (bool, optional): Whether words that contain numbers
            e.g., "3D" should be considered.
    
    Returns:
        list: Array of tokens.
    """
    if lowercase:
        matches = re.findall(r"\b[a-z]+\b", text.lower())
    else:
        matches = re.findall(r"\b[a-z]+\b", text)
    if not matches:
        return []
    return matches


def normalize_rows(M):
    return normalize_along_axis(M, 1)


def normalize_cols(M):
    return normalize_along_axis(M, 0)


def normalize_along_axis(M, axis):
    epsilon = np.finfo(float).eps
    norms = np.sqrt((M * M).sum(axis=axis, keepdims=True))
    norms += epsilon  # to avoid division by zero
    return M / norms

def get_elements(text):
    elements = []
    for paragraph in get_paragraphs(text):
        elements += get_sentences(paragraph)
    elements = [el.strip() for el in elements]
    elements = [el for el in elements if len(el) >= 3 and re.search("[A-Za-z]", el)]
    return elements
