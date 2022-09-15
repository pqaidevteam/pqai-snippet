"""
Reranking Service
"""
import os
import requests

RERANKING_SRV_ENDPOINT = os.environ.get("RERANKING_SRV_ENDPOINT")
assert isinstance(RERANKING_SRV_ENDPOINT, str)


class CustomRanker:
    """Runs CustomRanker model from PQAI Reranker"""
    endpoint = RERANKING_SRV_ENDPOINT

    @classmethod
    def rank(cls, query, docs):
        """The rank function takes a query and a list of documents as input. It returns the ranks of the documents in relation to the query.

        Args:
            cls: Used to Access the class variables.
            query (str): Pass the query string to the reranking service.
            docs (str): The documents to be ranked.
        Returns:
            list(): A list of ranks.
        """
        url = f"{cls.endpoint}/rerank"
        params = {"model": "custom-ranker", "query": query, "docs": docs}
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        ranks = response.json().get("ranks")
        return ranks


class ConceptMatchRanker:
    """Runs CustomMatchRanker model from PQAI Reranker"""
    endpoint = RERANKING_SRV_ENDPOINT

    @classmethod
    def score(cls, query, doc):
        """Takes a query and a document as input,and returns the score of the document for that query.

        Args:
            cls: Used to Access the class variables.
            query (str): Specify the query string.
            doc (str): The text of a document that is being scored.
        Returns:
             (float): The score of the document.
        """
        url = f"{cls.endpoint}/score"
        params = {"model": "concept-match-ranker", "query": query, "doc": doc}
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        score = response.json().get("score")
        return score

    @classmethod
    def rank(cls, query, docs):
        """The rank function takes a query and a list of documents as input.
        It returns the ranks of the documents in relation to the query.

        Args:
            cls: Used to Access the class variables of the class.
            query (str): Query string.
            docs (list): The list of documents that are to be reranked.

        Returns:
            A list of integers, which correspond to the rank of each document in the query.
        """
        url = f"{cls.endpoint}/rerank"
        params = {"model": "concept-match-ranker", "query": query, "docs": docs}
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        ranks = response.json().get("ranks")
        return ranks
