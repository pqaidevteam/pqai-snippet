"""
Utility Functions Module
"""
import re
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


def normalize_rows(M):
    """It takes a matrix and normalizes each row so that the sum of squares of each row is 1.
    For example, if the input is:
    [[3 4]
     [2 3]])
    Then the output should be:
    [[0.6 0.8]  # = [[3/5 4/5] [2/7 3/7]]^T as instructed in part 2 above.)

    Args:
        M (n-d Array): Used to Store the matrix that is being normalized.
    Returns:
         (n-d Array): A matrix with each row normalized to a length of 1.
    """
    return normalize_along_axis(M, 1)


def normalize_cols(M):
    """The normalize_cols function normalizes the columns of a given matrix.

    Args:
        M (n-d Array): Used to store the matrix that being normalized.
    Returns:
    (n-d Array): A matrix with each column normalized.
    """
    return normalize_along_axis(M, 0)


def normalize_along_axis(M, axis):
    """The normalize_along_axis function normalizes the rows of a matrix along an axis.

    Parameters:
        M (numpy array): The matrix to be normalized.

        axis (int): The dimension to normalize along, 0 for rows and 1 for columns.
    Returns:
        numpy array: A normalized version of the input matrix with respect to the specified dimension.
    """
    epsilon = np.finfo(float).eps
    norms = np.sqrt((M * M).sum(axis=axis, keepdims=True))
    norms += epsilon  # to avoid division by zero
    return M / norms


def get_elements(text):
    """The function first splits the text into paragraphs, then it splits each paragraph into sentences.

    Args:
    text (str): Used to Pass the text that should be parsed.
    Returns:
        (list): A list of strings containing the sentences in the text.
    """
    elements = []
    for paragraph in get_paragraphs(text):
        elements += get_sentences(paragraph)
    elements = [el.strip() for el in elements]
    elements = [el for el in elements if len(el) >= 3 and re.search("[A-Za-z]", el)]
    return elements
