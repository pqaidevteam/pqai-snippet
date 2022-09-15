"""
Snippet Module
"""
import re
from random import randrange
import numpy as np
from core import utils
from core.encoder_srv import encode
from core.reranking_srv import CustomRanker, ConceptMatchRanker
from core.utils import get_sentences
from core.sensible_span_extractor import SensibleSpanExtractor

ranker = CustomRanker()
conceptmatch_ranker = ConceptMatchRanker()

get_spans = SensibleSpanExtractor().return_ranked


class SnippetExtractor:
    """
    Extracts snippets from text
    """
    MIN_SNIPPET_LENGTH = 50

    @classmethod
    def extract_snippet(cls, query, text, htl_on=False):
        """Returns the most relevant snippet of text from the document

        Args:
            query (str): Query string
            text (str): Long text string
        Returns:
            str :The snippet of text that is most relevant to the query.
        """
        sents = cls._get_mappable_sentences(text)
        idx = ranker.rank(query, sents)[0]
        before = ""
        center = sents[idx]
        after = ""
        if idx > 0:
            before = cls._last_few_words(sents[idx - 1])
        if idx < len(sents) - 1:
            after = cls._first_few_words(sents[idx + 1])
        if htl_on:
            center = KeywordHighlighter.highlighter_fn(query, center)[0]
        snippet = before + center + after
        return snippet

    @classmethod
    def map(cls, query, text):
        """Takes a list of elements and a text, and returns the mappings between each element and the sentences in which it appears.

        Args:
            cls: Used to Pass the class of the object that is calling this function.
            query (str): Query String.
            text (str): Doc text for extraction.
        Returns:
           (list): A list of dictionaries of the mapping.
        """
        elements = utils.get_elements(query)
        sents = cls._get_mappable_sentences(text)

        A = utils.normalize_rows(np.array(encode(elements, "sbert")))
        B = utils.normalize_rows(np.array(encode(sents, "sbert")))

        cosine_sims = np.dot(A, B.T)
        sent_idxs = cosine_sims.argmax(axis=1)

        mappings = []
        for i, element in enumerate(elements):
            mapping = {}
            mapping["element"] = element
            try:
                multi_snippet = SubsentSnippetExtractor(element, text).extract()
                mapping["mapping"] = multi_snippet
                mapping["ctx_before"] = ""
                mapping["ctx_after"] = ""
                mapping["similarity"] = 0.0
            except Exception as e:
                traceback.print_exc()
                j = sent_idxs[i]
                mapping["mapping"] = sents[j]
                mapping["ctx_before"] = sents[j - 1] if j - 1 > 0 else ""
                mapping["ctx_after"] = sents[j + 1] if j + 1 < len(sents) else ""
                mapping["similarity"] = float(cosine_sims[i][j])
            mappings.append(mapping)

        return mappings

    @classmethod
    def _get_mappable_sentences(cls, text):
        """Takes a string and returns a list of sentences that are mappable.
        A sentence is mappable if it contains at least one word that has an entry in the dictionary.

        Args:
            cls: Used to Pass the class object.
            text (str): Used to Get the sentences from the text.
        Returns:
             (list): A list of sentences that are mappable.
        """
        sents = utils.get_sentences(text)
        sents = [s for s in sents if cls._is_mappable(s)]
        return sents

    @classmethod
    def _is_mappable(cls, sent):
        """Checks if a sentence is long enough to be considered
        a snippet, starts with an uppercase letter, and does not end with a colon.

        Args:
            cls: Used to Access the class variables.
            sent (str): The sentence query.
        Returns:
             (bool): A boolean value.
        """
        cond_1 = len(sent) >= cls.MIN_SNIPPET_LENGTH
        cond_2 = re.match("[A-Z]", sent)
        cond_3 = not sent.endswith(":")
        return True if (cond_1 and cond_2 and cond_3) else False

    @classmethod
    def _last_few_words(cls, sent):

        """
        Takes a sentence as input and returns the last few words of that sentence.
        The number of words returned is randomly chosen between 5 and 10.

        Args:
            cls: Used to Access the class variables.
            sent (str): The sentence query.
        Returns:
            (str):A string of the last few words in a sentence.
        """

        num_words = randrange(5, 10)
        i = len(sent) - 1
        n = 0
        while i > 0 and n < num_words:
            if sent[i] == " ":
                n += 1
            i -= 1
        return "..." + sent[i + 1 : len(sent)] + " "

    @classmethod
    def _first_few_words(cls, sent):
        """Takes a string and returns the first few words of that string.

        Args:
            cls: Used to Pass the class object to the function.
            sent (str): The sentence query..
        Returns:
            (str): The first few words.
        """
        num_words = randrange(2, 5)
        i = 0
        n = 0
        while i < len(sent) and n < num_words:
            if sent[i] == " ":
                n += 1
            i += 1
        return " " + sent[0:i] + "..."


class CombinationalMapping(SnippetExtractor):
    """Element mapping"""
    def __init__(self, query, texts):
        """The __init__ function is called when an instance of the class is created.

        Args:
            self: Used to Reference the object that is being created.
            query (str): The query string.
            texts (str): Store the texts that will be used for the summarization.
        Returns:
             The object itself.
        """
        self._texts = texts
        self._query = query
        self._elements = utils.get_elements(query)
        self._sents = [self._get_mappable_sentences(text) for text in texts]

    def map(self, table=False):
        """Takes a list of elements and returns a list of mapped elements.

        Args:
            self: Used to Reference the object of the class.
            table=False: Used to Return the mapping as a list of tuples.
        Returns:
             A list of the best matches for each element.
        """
        mapping = [
            self._select_best(self._map_element_with_all(el)) for el in self._elements
        ]
        if not table:
            return mapping
        return self._format_as_table(mapping)

    def _map_element_with_all(self, el):
        """Takes an element and maps it to a list of elements.

        Args:
            self: Used to Access the class attributes.
            el (str): The element to be mapped.
        Returns:
             A list of the results of mapping each text with the element.
        """
        n_texts = len(self._texts)
        return [self._map_element_with_ith(el, i) for i in range(n_texts)]

    def _map_element_with_ith(self, el, i):
        """Takes an element and a document index, and returns a dictionary containing the following keys:
            * element: The original element.
            * doc: The document index.
            * mapping: A sentence from the document that is most similar to the given
              concept.

        Args:
            self: Used to Reference the class object.
            el (str): The element to be mapped.
            i (str):  Select the sentence from which to extract the context.
        Returns:
             (dict): A dictionary with map.
        """
        sents = self._sents[i]
        k = conceptmatch_ranker.rank(el, sents)[0]
        dist = conceptmatch_ranker.score(el, sents[k])
        return {
            "element": el,
            "doc": i,
            "mapping": sents[k],
            "ctx_before": sents[k - 1] if k - 1 > 0 else "",
            "ctx_after": sents[k + 1] if k + 1 < len(sents) else "",
            "similarity": dist,
        }

    def _select_best(self, mappings):
        """Takes a list of mappings and returns the mapping with the highest similarity score.

        Args:
            self: Used to Reference the class object.
            mappings: Used to Store the similarity scores of each word in the text.
        Returns:
            The mapping with the highest similarity score.
        """
        return sorted(mappings, key=lambda x: x["similarity"])[0]

    def _format_as_table(self, elmaps):
        """Takes a list of dictionaries and formats them as a table.
        The first dictionary in the list is used as the header row, with keys from all other dictionaries used for data.

        Args:
            self: Used to Access the class attributes.
            elmaps (dict): The mapping of each element to its corresponding document.
        Returns:
            (list): A list of lists that can be used to create a table.
        """
        table = []
        header = ["Elements"] + list(range(len(self._texts)))
        table.append(header)
        for elmap in elmaps:
            row = []
            row.append(elmap["element"])
            for i in range(len(self._texts)):
                if elmap["doc"] == i:
                    row.append(elmap["mapping"])
                else:
                    row.append("")
            table.append(row)
        return table


class SubsentSnippetExtractor:
    """
    Extracts key phrases from text
    """
    def __init__(self, query, doc):
        """The __init__ function is called when an instance of the class is created.

        Args:
            self: Used to Refer to an instance of a class.
            query (str): the query string.
            doc (str): the document that is being ranked.
        Returns:
             The object itself.
        """
        self.query = query
        self.doc = doc

    def extract(self):
        """Takes a document as input and returns the keyphrases found in that document.

        Args:
            self: Used to Access the variables and methods of the class in python.
        Returns:
             A string of the keyphrases joined by a space.
        """
        keyphrases = self._find_keyphrases_in_doc()
        subs = self._get_spliced_subsent_snippets(keyphrases)
        return self._join(subs)

    def _find_keyphrases_in_doc(self):
        """Extracts the concepts from the query and document using the _extract_concepts function.

        Args:
            self: Used to Access the attributes and methods of the class in python.
        Returns:
             (list): A list of the keyphrases that are found in the document.
        """
        query_concepts = self._extract_concepts(self.query)
        doc_concepts = self._extract_concepts(self.doc)
        keyphrases = []
        for concept in query_concepts:
            i = conceptmatch_ranker.rank(concept, doc_concepts)[0]
            keyphrases.append(doc_concepts[i])
        return keyphrases

    def _get_spliced_subsent_snippets(self, matches):
        """Takes in a list of matches and returns a list of snippets that contain the  matches.

        Args:
            self: Used to Reference the class instance.
            matches: the matches found in the text.
        Returns:
             (list): A list of the snippets that contain the matches.
        """
        sents = get_sentences(self.doc)
        sents = [s for s in sents if not re.search(r"\d{2,}", s)]

        sent_scores = [sum([1 for m in matches if m in s]) for s in sents]
        sent_ranked = [
            sents[i] for i in np.argsort(sent_scores)[::-1] if sent_scores[i] > 0
        ]
        temp_str = ""
        subs = []

        for match in matches:
            if match in temp_str:
                continue

            for sent in sent_ranked:
                if match in sent:
                    sub = self._span_containing(match, sent)
                    temp_str += " ..." + sub + "..."
                    subs.append(sub)
                    break
        return subs

    def _extract_concepts(self, text):
        """Extracts concepts from a given text.

        Args:
            self: Used to Reference the class object itself.
            text(str): Text string.
        Returns:
             (list): A list of the concepts found in a text.
        """

        target_concepts = set()
        for sent in get_sentences(text):
            for c in encode(sent, "boe"):
                target_concepts.add(c)
        return list(target_concepts)

    def _span_containing(self, entity, sent):
        """Takes in an entity and a sentence. It returns the span of that sentence that contains the entity.

        Args:
            self: Used to Reference the class itself.
            entity (str): Search entity.
            sent (str): The sentence in which the entity is present.
        Returns:
            A span of the sentence that contains the entity.
        """
        spans = get_spans(sent)
        for span in spans:
            if entity in span:
                return span

    def _join(self, subs):
        """joins a list of strings together with an ellipsis and a trailing space.

        Args:
            self: Used to Reference the object itself.
            subs (list): The list of substrings to be joined.
        Returns:
            (str): A string that is the concatenation of the strings in subs.
        """
        return "..." + "... ".join(subs) + "..."
