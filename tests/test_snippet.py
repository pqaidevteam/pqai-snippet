import os
import unittest
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Run tests without using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
sys.path.append(BASE_DIR)

ENV_FILE = f"{BASE_DIR}/.env"
load_dotenv(ENV_FILE)

from core.snippet import (
	SnippetExtractor,
	CombinationalMapping,
	SubsentSnippetExtractor
)

class TestSnippetExtractor(unittest.TestCase):

	def setUp(self):
		self.query = 'fluid formation sampling'
		self.longquery = 'A method of sampling formation fluids. The method includes lowering a sampling apparatus into a borewell.'
		self.text = 'A fluid sampling system retrieves a formation fluid sample from a formation surrounding a wellbore extending along a wellbore axis, wherein the formation has a virgin fluid and a contaminated fluid therein. The system includes a sample inlet, a first guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a first direction along the wellbore axis, and a second guard inlet positioned adjacent to the sample inlet and spaced from the sample inlet in a second, opposite direction along the wellbore axis. At least one cleanup flowline is fluidly connected to the first and second guard inlets for passing contaminated fluid, and an evaluation flowline is fluidly connected to the sample inlet for collecting virgin fluid.'

	def test__can_create_snippet(self):
		snip = SnippetExtractor.extract_snippet(self.query, self.text)
		self.assertIsInstance(snip, str)
		self.assertGreater(len(snip), 10)

	def test__can_do_mapping(self):
		mapping = SnippetExtractor.map(self.longquery, self.text)
		self.assertIsInstance(mapping, list)
		self.assertGreater(len(mapping), 1)


class TestCombinationalMapping(unittest.TestCase):

	def setUp(self):
		self.query = "A method of formation fluid sampling comprising:\nlowering a probe into a borewell."
		self.docs = []
		for pn in ["US7654321B2", "US7654322B2"]:
			with open(f"{TEST_DIR}/files/{pn}.json", "r") as f:
				patent = json.load(f)
				doc = patent.get("description")
				self.docs.append(doc)

	def test__can_map(self):
		mapping = CombinationalMapping(self.query, self.docs).map()
		self.assertIsInstance(mapping, list)
		self.assertEqual(len(mapping), 2)
		for row in mapping:
			self.assertIsInstance(row, dict)
			self.assertIn("element", row)
			self.assertIn("doc", row)
			self.assertIn("mapping", row)
			self.assertIn("ctx_before", row)
			self.assertIn("ctx_after", row)
			self.assertIn("similarity", row)

	def test__can_map_as_a_table(self):
		mapping = CombinationalMapping(self.query, self.docs).map(table=True)
		self.assertIsInstance(mapping, list)
		self.assertEqual(len(mapping), 3)
		for row in mapping:
			self.assertIsInstance(row, list)
			self.assertEqual(len(row), 3)


class TestSubsentSnippetExtractor(unittest.TestCase):

	def setUp(self):
		self.query = "A method of formation fluid sampling comprising:\nlowering a probe into a borewell."
		with open(f"{TEST_DIR}/files/US7654321B2.json", "r") as f:
			patent = json.load(f)
			self.doc = patent.get("description")

	@unittest.skip("temp")
	def test__can_extract_snippet(self):
		snippet = SubsentSnippetExtractor(self.query, self.doc).extract()
		self.assertIsInstance(snippet, str)
		self.assertGreater(len(snippet), 0)


if __name__ == '__main__':
	unittest.main()
