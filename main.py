import os
import dotenv
import uvicorn
from fastapi import FastAPI
dotenv.load_dotenv()

from core.snippet import SnippetExtractor

app = FastAPI()


@app.get('/snippet')
async def extract_snippet(query, doc):
    """ Extracts snippet from long text.

    Args:
        query (str): Query
        doc (str): Text from which snippet is extracted
    Returns:
        str: Snippet
    """
    snippet = SnippetExtractor.extract_snippet(query, doc)
    return snippet

@app.get('/mapping')
async def create_mapping(query, doc):
    """ Extracts snippet from long text.

    Args:
        query (str): Query
        doc (str): Text from which snippet is extracted
    Returns:
        str: Snippet
    """
    mapping = SnippetExtractor.map(query, doc)
    return mapping

if __name__ == "__main__":
    port = int(os.environ['PORT'])
    uvicorn.run(app, host="127.0.0.1", port=port)




