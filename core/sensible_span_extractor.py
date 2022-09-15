"""
Sensible Span Extractor module
"""
import re
import json
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

K.set_image_data_format("channels_last")

BASE_DIR = str(Path(__file__).parent.parent.resolve())
models_dir = "{}/assets/".format(BASE_DIR)


class SensibleSpanExtractor:
    """Extract meaningful span of words from a sentence"""
    vocab_dict_file = os.path.join(models_dir, "span_extractor_dictionary.json")
    vocab_file = os.path.join(models_dir, "span_extractor_vocab.json")
    model_file = os.path.join(models_dir, "span_extractor_model.hdf5")
    embeddings_file = os.path.join(models_dir, "span_extractor_vectors.txt")

    def __init__(self):
        """The __init__ function is called when an instance of the class is created."""
        self._model = None
        self._dims = 16
        self._vocab_size = 844
        self._dict = {}
        self._embs = {}
        self._emb_matrix = np.empty((self._vocab_size, self._dims))
        self._vocab_map = {}
        self._punct_map = {
            "!": "<exclm>",
            '"': "<dinvc>",
            "#": "<hash>",
            "$": "<dlr>",
            "%": "<pcnt>",
            "&": "<and>",
            "'": "<sinvc>",
            "(": "<lb>",
            ")": "<rb>",
            "*": "<astk>",
            "+": "<plus>",
            ",": "<coma>",
            "-": "<minus>",
            ".": "<fstp>",
            "/": "<fslsh>",
            ":": "<cln>",
            ";": "<scln>",
            "<": "<lt>",
            "=": "<eq>",
            ">": "<gt>",
            "?": "<qm>",
            "@": "<arte>",
            "[": "<lsb>",
            "\\": "<bslsh>",
            "]": "<rsb>",
            "^": "<rtp>",
            "_": "<uscr>",
            "`": "<btck>",
            "{": "<lcb>",
            "|": "<pipe>",
            "}": "<rcb>",
            "~": "<tlde>",
            "“": "sdinvc",
            "”": "edinvc",
        }
        self.chars = "abcdefghijklmnopqrstuvwxyz"
        self.MIN_LEN = 5
        self.MAX_LEN = 10

        self._load_model()
        self._load_dict()
        self._load_embeddings()
        self._load_embedding_matrix()
        self._load_vocab_map()

    def _load_model(self):
        """Loads the model from the model file"""
        model = load_model(self.model_file)
        self._model = Model(model.input[:2], model.layers[-3].output)

    def _load_dict(self):
        """Loads the dictionary from the vocab_dict.json file into a Python dictionary object"""
        with open(self.vocab_dict_file, "r") as file:
            self._dict = json.load(file)

    def _load_embeddings(self):
        """Loads the embeddings from a file and stores them in a dictionary.
        The function also adds the padding and unknown vectors to the dictionary.
        """
        with open(self.embeddings_file, "r") as file:
            lines = file.read().strip().splitlines()
        for line in lines:
            token, *vector = line.strip().split()
            self._embs[token] = np.array(vector, dtype="float32")
        self._include_pad_unk_vectors()

    def _include_pad_unk_vectors(self):
        """Adds a vector for the <pad> and <unk> tokens to the embedding matrix."""
        low = -0.00001
        high = 0.00001
        self._embs["<pad>"] = [np.random.uniform(low, high) for _ in range(self._dims)]
        self._embs["<unk>"] = self._embs["<raw_unk>"]
        self._embs.pop("<raw_unk>")

    def _load_embedding_matrix(self):
        """Loads the embedding matrix for the model."""
        for word, i in self._dict.items():
            self._emb_matrix[i] = self._embs[word]

    def _load_vocab_map(self):
        """Loads the vocabulary file into a dictionary."""
        with open(self.vocab_file, "r") as fd:
            vocab = json.load(fd)
            self._vocab_map = {word: True for word in vocab}

    def extract_from(self, sentence):
        """Returns the most relevant  sentence from the text.

        Args:
            self: Used to Access the class attributes.
            sentence (str): The sentence to be processed.
        Returns:
            (str): The first candidate sentence with the highest score.t
        """
        candidates = self._encode_for_nn(sentence)
        i = self._rank(candidates)[0]
        return self._strip_punctuations(" ".join(candidates[0][i]))

    @lru_cache(maxsize=50000)
    def return_ranked(self, sentence):
        """The return_ranked function takes a sentence as input and returns the top 10 ranked candidate answers.

        Args:
            self: Used to Access the class attributes.
            sentence (str): Query Sentence.
        Returns:
            (list): The ranked list of spans for a given sentence.
        """
        candidates = self._encode_for_nn(sentence)
        ns = self._rank(candidates)
        spans = [self._strip_punctuations(" ".join(candidates[0][n])) for n in ns]
        spans = [s for s in spans if self._passes_post_filter(s)]
        return spans

    def _encode_for_nn(self, sentence):
        """Takes a sentence as input and returns three arrays: tokens, chargrams, and word vectors.

        Args:
            self: Used to Access the class attributes.
            sentence (str): The sentence to be encoded.
        Returns:
            Three arrays.
        """
        cased = self._tokenize(sentence, lower=False)
        uncased = [t.lower() for t in cased]
        limits = [self.MIN_LEN, self.MAX_LEN]
        tokens = SubsequenceExtractor(cased).extract(*limits)
        spans = SubsequenceExtractor(uncased).extract(*limits)
        chargrams = [self._span2chargram(span) for span in spans]
        word_vectors = [self._embed_words(span) for span in spans]
        return tokens, np.array(chargrams), np.array(word_vectors)

    def _rank(self, candidates):
        """Takes in a list of candidates and returns the indices of the candidates sorted by their rank. The higher the score, the better it is.

        Args:
            self: Used to Access the attributes of the class.
            candidates (list): the candidates to be ranked.
        Returns:
            (list): The predicted probability of each candidate.
        """
        tokens, X_chargrams, X_word_vectors = candidates
        pred = self._model.predict(
            [X_word_vectors, X_chargrams], batch_size=256
        ).flatten()
        pred = K.softmax(pred)
        return np.argsort(pred)[::-1]

    def _passes_post_filter(self, span):
        """A helper function that is used to filter out spans of text that are not of interest.

        Args:
            self: Used to Reference the class instance.
            span: Used to Check if the span contains a parenthesis.
        Returns:
            (bool): A boolean value.

        """
        if ")" in span and "(" not in span:
            return False
        if "(" in span and ")" not in span:
            return False
        return True

    def _tokenize(self, sentence, lower=True):
        """Splits a sentence into tokens.

        Args:
            self: Used to Reference the class object.
            sentence (str): The sentence that needs to be tokenized.
            lower=True: Convert the text to lower case.
        Returns:
            (list): A list of tokens.
        """
        sentence = sentence.lower() if lower else sentence
        tokens = re.findall(r"(\w+|\W+)", sentence)
        tokens = [t for t in tokens if t.strip()]
        return tokens

    def _span2chargram(self, span):
        """Takes a span of words and converts it to its character-level ngram representation.

        Args:
            self: Used to Access the class variables.
            span: Used to Get the span of words in a sentence.
        Returns:
            (list): A list of lists, where each sublist contains the character n-grams for a token in the span.
        """
        span_len = min(len(span), self.MAX_LEN)
        span_chargram_unpadded = [self._word2chargram(span[i]) for i in range(span_len)]
        span_chargram = self._padding_int_arr(span_chargram_unpadded)
        return span_chargram

    def _word2chargram(self, word):
        """Takes a word and returns the 20-dimensional chargram representation of that word.

        Args:
            self: Used to Reference the class object.
         word(str): The word query.
        Returns:
            (list): A list of numbers that represent the character n-grams of a word.
        """

        chargrams = [0] * 20
        char_list = list(word)
        for i in range(min(len(char_list), 20)):
            if char_list[i] in self.chars:
                chargrams[i] = self.chars.index(char_list[i]) + 1
        return chargrams

    def _embed_words(self, span):
        """Takes a span of words and returns the corresponding embedding matrix.

        Args:
            self: Used to Access the class attributes.
            span: The span of words.
        Returns:
            (array): A vector of shape (max_len, embedding_size).
        """
        span_len = min(len(span), self.MAX_LEN)
        span_int_array = self._to_int_array(span)
        span_word_emb_unpadded = [
            self._emb_matrix[span_int_array[i]] for i in range(span_len)
        ]
        span_word_emb = self._padding_int_arr(span_word_emb_unpadded)
        return span_word_emb

    def _get_masked_tokens(self, sent_tokens):
        """Takes in a list of tokens and returns a new list where all the numbers are replaced with <num> and all non-alphanumeric characters are replaced with <alphanum>. If the token is not in the vocabulary, it is replaced with <unk>.

        Args:
            self: Used to Reference the class instance.
            sent_tokens (list):Tthe list of tokens.
        Returns:
            (list): A list of masked tokens.
        """
        masked_sent_tokens = []
        for token in sent_tokens:
            if self._is_number(token):
                masked_sent_tokens.append("<num>")
            elif self._is_alphanumeric(token, "fast"):
                masked_sent_tokens.append("<alphanum>")
            elif token in self._punct_map:
                masked_sent_tokens.append(self._punct_map[token])
            elif token not in self._vocab_map:
                masked_sent_tokens.append("<unk>")
            else:
                masked_sent_tokens.append(token)
        return masked_sent_tokens

    def _is_alphanumeric(self, token, mode="slow"):
        """Checks if a token is alphanumeric. If the token is not alphanumeric, it checks if the mode is slow and then checks to see if it's a number. If all of these fail, return False.

        Args:
            self: Used to Reference the class instance.
            token (list): The list of tokens.
            mode="slow": Used to Determine whether the function should be used to check if a token is alphanumeric or numeric.
        Returns:
            (bool): True if the token is alphanumeric and false otherwise.
        """
        if token.isalnum():
            if not token.isalpha():
                if mode == "slow":
                    if is_number(token):
                        return False
                return True
        return False

    def _is_number(self, token):
        """Checks if the token is a number. It returns True if it is, and False otherwise.

        Args:
            self: Used to Access variables that belongs to the class.
            token (list): The list of tokens.
        Returns:
            (bool): A boolean value.
        """
        return bool(re.search(r"^\d+(\.\d+)?$", token))

    def _encode_tokens(self, tokens):
        """Encodes the tokens in a list of strings into a list of integers.

        Args:
            self: Used to Access the variables and methods of the class in python.
             tokens (list): The list of tokens.
        Returns:
            (list): The list of the tokens in the input string, encoded as a list of integers.
        """
        encoded_tokens = []
        for token in tokens:
            default = self._dict["<unk>"]
            encoded_tokens.append(self._dict.get(token, default))
        return encoded_tokens

    def _to_int_array(self, sent_tokens):
        """Takes a list of tokens and encodes them as integers.

        Args:
            self: Used to Reference the class object.
            sent_tokens (list): The tokens of the sentence that is to be encoded.
        Returns:
            (list): The encoded tokens.
        """
        masked_sent_tokens = self._get_masked_tokens(sent_tokens)
        encoded_tokens = self._encode_tokens(masked_sent_tokens)
        return encoded_tokens

    def _get_pad_token(self, int_arr):
        """A helper function that is used to pad the input data with zeros.
        It takes in an array of integers and returns a padded array of integers.

        Args:
            self: Used to Access the class attributes.
            int_arr: Used to Get the length of the input array.
        Returns:
            (array): The first row of the embedding matrix.
        """

        if len(int_arr[0]) == 16:
            pad_token = self._emb_matrix[0]
        else:
            pad_token = self._word2chargram("0")
        return pad_token

    def _padding_int_arr(self, int_arr):
        """Takes in an array of integers and pads it with the pad token until it is a length of 30.
        The function returns the padded array.

        Args:
            self: Used to Access the attributes and methods of the class in python.
             int_arr: the integer representation of each word in a sentence.
        Returns:
            (array): The int_arr with the pad token appended to it until it is of length 30.
        """

        pad_token = self._get_pad_token(int_arr)
        for j in range(30 - len(int_arr)):
            int_arr.append(pad_token)
        int_arr = np.array(int_arr)
        return int_arr

    def _strip_punctuations(self, text):
        """Removes punctuations from the text.

        Args:
            self: Used to Reference the class object.
            text(str): The text to be cleaned.
        Returns:
            (str): The text without the punctuations.
        """
        patterns = [
            r"\s([\!\%\)\,\.\:\;\?\]\}\”])",  # no space before these symbols
            r"([\"\#\$\(\@\[\\\{\“])\s",  # no space after these
            r"\s([\'\*\+\-\/\<\=\>\^\_\`\|\~])\s",
        ]  # no space before/after
        for pattern in patterns:
            text = re.sub(pattern, r"\1", text)
        text = re.sub("  ", " ", text)  # remove double spaces
        return text


class SubsequenceExtractor:
    """Extract subsequence of specified length from text"""
    def __init__(self, sequence):
        """The __init__ function is called when an instance of the class is created."""
        self._seq = sequence
        self._seqlen = len(sequence)

    def extract(self, minlen, maxlen=None):
        """Returns all subsequences of the given lengths in the sequence, including
        the empty sequence. The subsequences are returned in order from shortest to
        longest.

        Args:
            self: Used to Reference the object itself.
            minlen (int): The minimum length of subsequence that will be extracted.
            maxlen=None (int): The subsequences can be of any length.
        Returns:
            (list): A list of all possible subsequences in the sequence.
        """
        maxlen = minlen if maxlen is None else maxlen
        subsequences = []
        for L in self._possible_lengths(minlen, maxlen):
            subsequences += self._get_subsequences_of_length(L)
        return subsequences

    def _possible_lengths(self, minlen, maxlen):
        """A helper function that returns a list of possible lengths for the sequence.

        Args:
            self: Used to Reference the object of the class.
             minlen: the minimum length of a sequence.
             maxlen: Limit the length of the sequences that are returned.
        Returns:
            A list of possible lengths for the sequence.
        """
        if self._seqlen <= minlen:
            return [self._seqlen]
        if minlen < self._seqlen <= maxlen:
            return list(range(minlen, self._seqlen + 1))
        return list(range(minlen, maxlen + 1))

    def _get_subsequences_of_length(self, L):
        """ Returns a list of all subsequences of length L that can be extracted from the sequence.

        Args:
            L (int): The length of subsequence to extract. Must be greater than 0 and less than or equal to the length of the sequence.

        Returns:
            subsequences (list): A list containing all subsequences that can be extracted from the sequence with lengths L.
        """
        if L == 0:
            return []
        start_positions = range(self._seqlen - L + 1)
        return [self._seq[p : p + L] for p in start_positions]
