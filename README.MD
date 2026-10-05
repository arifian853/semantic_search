# Text Embeddings & Semantic Search

A small Python project for learning text embeddings with Sentence Transformers. It uses Indonesian examples to compare sentences and find documents by meaning.

Both scripts use the multilingual model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Examples

- **`app.py`** — Generate a text embedding, inspect its shape, and compare two sentences using cosine similarity and Euclidean distance.
- **`semantic_search.py`** — Compare a query against five sample documents and print the most relevant document with its similarity score.

## Setup

Install Python, then run these commands from the project folder:

```sh
python -m venv venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install sentence-transformers numpy
```

The first run downloads the model, so an internet connection is required.

## Usage

Run the semantic search example:

```sh
python semantic_search.py
```

The default query is `Hewan yang termasuk mamalia`, which searches for a document about mammals. Edit `query` and `documents` in the script to try your own examples.

To run the embedding comparison:

```sh
python app.py
```

**Before running `app.py`:** add `#` before the line `Melihat bentuk embedding dari sebuah teks` so Python treats it as a comment. Edit `text`, `text1`, and `text2` to experiment with other sentences.
