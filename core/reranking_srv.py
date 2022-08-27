import os
import requests

RERANKING_SRV_ENDPOINT = os.environ.get("RERANKING_SRV_ENDPOINT")
assert isinstance(RERANKING_SRV_ENDPOINT, str)


class CustomRanker:

    endpoint = RERANKING_SRV_ENDPOINT
    
    @classmethod
    def rank(cls, query, docs):
        url = f"{cls.endpoint}/rerank"
        params = {
            "model": "custom-ranker",
            "query": query,
            "docs": docs
        }
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        ranks = response.json().get("ranks")
        return ranks


class ConceptMatchRanker:

    endpoint = RERANKING_SRV_ENDPOINT
    
    @classmethod
    def score(cls, query, doc):
        url = f"{cls.endpoint}/score"
        params = {
            "model": "concept-match-ranker",
            "query": query,
            "doc": doc
        }
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        score = response.json().get("score")
        return score

    @classmethod
    def rank(cls, query, docs):
        url = f"{cls.endpoint}/rerank"
        params = {
            "model": "concept-match-ranker",
            "query": query,
            "docs": docs
        }
        response = requests.post(url, json=params)
        if response.status_code != 200:
            raise Exception("Reranking service failure")
        ranks = response.json().get("ranks")
        return ranks
