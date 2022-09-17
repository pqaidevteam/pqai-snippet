import os
import unittest
import sys
from pathlib import Path
from dotenv import load_dotenv

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
sys.path.append(BASE_DIR)
ENV_FILE = f"{BASE_DIR}/.env"

load_dotenv(ENV_FILE)

from core.encoder_srv import encode
from core.reranking_srv import CustomRanker,  ConceptMatchRanker

ENCODER_SRV_ENDPOINT = os.environ.get("ENCODER_SRV_ENDPOINT")
RERANKING_SRV_ENDPOINT = os.environ.get("RERANKING_SRV_ENDPOINT")

class TestEncoderAPI(unittest.TestCase):
    """ Enocder Service API test class"""

    def test_checkendpoint(self):
        self.assertIsInstance(ENCODER_SRV_ENDPOINT, str)

    def test_can_enocde(self):
        sent = "base station and mobile station"
        expected = ["base station", "mobile station"]
        result = encode(sent, "boe")
        self.assertEqual(result, expected)

    def test__throws_error_when_wrong_encoder_specified(self):
        text = "Some random text"
        self.assertRaises(Exception, encode, text, "random")

class TestRerankingAPI(unittest.TestCase):
    """Reranking Service API test class"""

    def test_checkendpoint(self):
        self.assertIsInstance(RERANKING_SRV_ENDPOINT, str)

    def test_throws_error_when_wrong_arguments_specified(self):
        query = 123
        docs = []
        self.assertRaises(Exception, CustomRanker.rank, query, docs)
        self.assertRaises(Exception, ConceptMatchRanker.rank, query, docs)
        self.assertRaises(Exception, ConceptMatchRanker.score, query, docs)




if __name__ == "__main__":
    unittest.main()
