"""Test for service API
Attributes:
    dotenv_file (str): Absolute path to .env file (used for reading port no.)
    HOST (str): IP address of the host where service is running
    PORT (str): Port no. on which the server is listening
    PROTOCOL (str): `http` or `https`
"""
import unittest
import os
import json
import re
import socket
from pathlib import Path
import requests
import numpy as np
from dotenv import load_dotenv

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
ENV_FILE = f"{BASE_DIR}/.env"
load_dotenv(ENV_FILE)

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_not_running = sock.connect_ex((HOST, int(PORT))) != 0

if server_not_running:
    print("Server is not running. API tests will be skipped.")


@unittest.skipIf(server_not_running, "Works only when true")
class TestAPI(unittest.TestCase):

    """Check operation of all API routes"""

    def test__can_extract_snippet(self):
        """Check if a valid response is returned for a legit request"""
        query = 'fluid formation sampling'
        doc = 'A fluid sampling system retrieves a formation fluid sample from a formation surrounding a wellbore extending along a wellbore axis, wherein the formation has a virgin fluid and a contaminated fluid therein. The system includes a sample inlet, a first guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a first direction along the wellbore axis, and a second guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a second, opposite direction along the wellbore axis. At least one cleanup flowline is fluidly connected to the first and second guard inlets for passing contaminated fluid, and an evaluation flowline is fluidly connected to the sample inlet for collecting virgin fluid.'
        params = {
            "query": query,
            "doc": doc
        }
        response = self.call_route("/snippet", params)
        self.assertEqual(200, response.status_code)

    def test__can_create_mapping(self):
        query = 'A method of sampling formation fluids. The method includes lowering a sampling apparatus into a borewell.'
        doc = 'A fluid sampling system retrieves a formation fluid sample from a formation surrounding a wellbore extending along a wellbore axis, wherein the formation has a virgin fluid and a contaminated fluid therein. The system includes a sample inlet, a first guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a first direction along the wellbore axis, and a second guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a second, opposite direction along the wellbore axis. At least one cleanup flowline is fluidly connected to the first and second guard inlets for passing contaminated fluid, and an evaluation flowline is fluidly connected to the sample inlet for collecting virgin fluid.'
        params = {
            "query": query,
            "doc": doc
        }
        response = self.call_route("/snippet", params)
        self.assertEqual(200, response.status_code)

    def call_route(self, route, params):
        """Make a request to given route with given parameters
        Args:
            route (str): Route, e.g. '/snippet'
            params (dict): Query string parameters
        Returns:
            requests.models.Response: Response against HTTP request
        """
        route = route.lstrip("/")
        url = f"{PROTOCOL}://{HOST}:{PORT}/{route}"
        response = requests.get(url, params)
        return response


if __name__ == "__main__":
    unittest.main()