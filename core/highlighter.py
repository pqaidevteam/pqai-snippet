"""
Highlighter module
"""
import re
import numpy as np
from core.utils import is_generic
from core.encoder_srv import encode


def highlight(query, text):
    """Given a query, highlight some relevant words in the given text
    snippet which would help user judge the relevancy of the snippet for
    the given query.

    Args:
        query (str): Query
        text (str): Snippet of text.

    Returns:
        str: Snippet with relevant words wrapped in <strong></strong>
            tags (to show as html).
    """
    words = list(set(re.findall(r"[a-z]+", query.lower())))
    terms = list(set(re.findall(r"[a-z]+", text.lower())))

    words = [word for word in words if not is_generic(word)]
    terms = [term for term in terms if not is_generic(term)]

    qvecs = encode(words, "sif")
    tvecs = encode(terms, "sif")

    qvecs = qvecs / np.linalg.norm(qvecs, ord=2, axis=1, keepdims=True)
    tvecs = tvecs / np.linalg.norm(tvecs, ord=2, axis=1, keepdims=True)

    sims = np.matmul(qvecs, tvecs.transpose())
    to_highlight = []
    for i in range(sims.shape[0]):
        j = np.argmax(sims[i])
        if sims[i][j] > 0.6:
            to_highlight.append(terms[j])

    replacement = "<dGn9zx>\\1</dGn9zx>"
    for term in to_highlight:
        pattern = r"\b(" + term + r")\b"
        text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

    while True:
        flag = False
        matches = re.findall(r"([a-z]+)\s\<dGn9zx\>", text, re.IGNORECASE)
        for match in matches:
            if not is_generic(match.lower()):
                flag = True
                pattern = match + " <dGn9zx>"
                replacement = "<dGn9zx>" + match + " "
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if not flag:
            break

    while True:
        flag = False
        matches = re.findall(r"\<\/dGn9zx\>\s([a-z]+)", text, re.IGNORECASE)
        for match in matches:
            if not is_generic(match.lower()):
                flag = True
                pattern = "</dGn9zx> " + match
                replacement = " " + match + "</dGn9zx>"
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if not flag:
            break

    pattern = r"dGn9zx"
    replacement = "strong"
    text = re.sub(pattern, replacement, text)
    return (text, to_highlight)
