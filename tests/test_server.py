"""Test for service API
"""
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).parent.parent.resolve()
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE.as_posix())
sys.path.append(BASE_DIR.as_posix())

from main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test__can_extract_snippet(self):
        """Check if a valid response is returned for a legit request"""
        query = 'fluid formation sampling'
        doc = 'A fluid sampling system retrieves a formation fluid sample from a formation surrounding a wellbore extending along a wellbore axis, wherein the formation has a virgin fluid and a contaminated fluid therein. The system includes a sample inlet, a first guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a first direction along the wellbore axis, and a second guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a second, opposite direction along the wellbore axis. At least one cleanup flowline is fluidly connected to the first and second guard inlets for passing contaminated fluid, and an evaluation flowline is fluidly connected to the sample inlet for collecting virgin fluid.'
        params = {
            "query": query,
            "doc": doc
        }
        response = self.client.get("/snippet", params=params)
        self.assertEqual(200, response.status_code)

    def test__can_create_mapping(self):
        query = 'A method of sampling formation fluids. The method includes lowering a sampling apparatus into a borewell.'
        doc = 'A fluid sampling system retrieves a formation fluid sample from a formation surrounding a wellbore extending along a wellbore axis, wherein the formation has a virgin fluid and a contaminated fluid therein. The system includes a sample inlet, a first guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a first direction along the wellbore axis, and a second guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a second, opposite direction along the wellbore axis. At least one cleanup flowline is fluidly connected to the first and second guard inlets for passing contaminated fluid, and an evaluation flowline is fluidly connected to the sample inlet for collecting virgin fluid.'
        params = {
            "query": query,
            "doc": doc
        }
        response = self.client.get("/snippet", params=params)
        self.assertEqual(200, response.status_code)

if __name__ == "__main__":
    unittest.main()
