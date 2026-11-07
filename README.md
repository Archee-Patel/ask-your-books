# Ask-Your-Books

A vintage-inspired web and command-line RAG application for asking questions
about PDF books. It extracts text page by page, chunks it with overlap, embeds
the chunks with Sentence Transformers, stores them in Chroma, and uses an LLM
to answer with page-level citations. When retrieved passages do not appear
relevant, it replies:

> I don't know based on the provided documents.

## Requirements

- Python 3.9 or newer
- A Groq API key for answer generation
- Internet access the first time the embedding model is downloaded

## Setup

```bash
git clone https://github.com/Archee-Patel/ask-your-books.git
cd ask-your-books
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`:

```dotenv
GROQ_API_KEY=your-groq-api-key
```

Edit `.env` or export variables to change the PDF folder, embedding model,
Groq answer model, top-k, or relevance threshold. The defaults use `data/books`,
`sentence-transformers/all-MiniLM-L6-v2`, Groq's `openai/gpt-oss-20b`,
five retrieved chunks, and a cosine-distance threshold of `0.55`. Check
[Groq's current model catalog](https://console.groq.com/docs/models) if a model
is unavailable to your account. The app uses
Groq's OpenAI-compatible API endpoint for chat completion.

## Sample books

Five reflowed, paginated sample PDFs are in `data/books/`:

- *Bulfinch's Mythology* — Thomas Bulfinch, Project Gutenberg ebook
  [#4928](https://www.gutenberg.org/ebooks/4928)
- *Myths and Legends of Ancient Greece and Rome* — E. M. Berens, ebook
  [#22381](https://www.gutenberg.org/ebooks/22381)
- *The Heroes; Or, Greek Fairy Tales for My Children* — Charles Kingsley,
  ebook [#677](https://www.gutenberg.org/ebooks/677)
- *Astrology: How to Make and Read Your Own Horoscope* — Sepharial, ebook
  [#46963](https://www.gutenberg.org/ebooks/46963)
- *Ptolemy's Tetrabiblos* — Claudius Ptolemy, ebook
  [#70850](https://www.gutenberg.org/ebooks/70850)

The sample PDFs are formatted from Project Gutenberg plain-text transcriptions;
their page numbers refer to the included PDF editions, not to another print or
ebook edition. They are intended as public-domain US samples. Copyright status
can differ by jurisdiction; check local law before redistribution. The
Project Gutenberg name is used only to identify the source, and this project
is not affiliated with Project Gutenberg.

To rebuild the samples from the source texts:

```bash
python -m pip install "reportlab>=4.1,<5.0"
python scripts/build_sample_pdfs.py
```

## Ingest PDFs

From the repository root, index the included books:

```bash
python -m ask_your_books.cli ingest
```

Add your own PDFs to `data/books/` and run the ingest command again. Or index a
different folder:

```bash
python -m ask_your_books.cli ingest --pdf-dir /path/to/pdfs
```

Each extracted chunk stores its relative PDF filename and 1-based PDF page
number. Chunks are limited to one PDF page so citations remain page-accurate.
Re-ingestion updates changed chunks instead of accumulating duplicates.

## Ask questions

Start the graphical reading room:

```bash
python -m ask_your_books.web
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. From the
reading room, ask questions, open the PDF shelf, or use **Update the catalog**
to index the books in your configured PDF folder.

The original command-line interface is still available:

```bash
python -m ask_your_books.cli ask "Who helped Theseus escape the Labyrinth?"
python -m ask_your_books.cli ask "What physical traits does Sepharial associate with Mars?"
python -m ask_your_books.cli ask "What is the capital of Iceland?"
```

Expected citations for the included PDF editions:

- “Who helped Theseus escape the Labyrinth?” — Ariadne helped him; citation:
  `[Bulfinch-s-Mythology.pdf, p. 78]`
- “What physical traits does Sepharial associate with Mars?” — The guide
  associates Mars with a strong, muscular, athletic build and a ruddy
  complexion; citation:
  `[Astrology-How-to-Make-and-Read-Your-Own-Horoscope.pdf, p. 8]`
- “What is the capital of Iceland?” — This is not answered by the sample
  books, so the app should say it does not know based on the provided documents.

## How it works

1. `pypdf` extracts text one page at a time.
2. Text is split into 220-word chunks with 40 words of overlap.
3. Sentence Transformers creates normalized embeddings.
4. Chroma stores embeddings, chunk text, filename, and 1-based page number.
5. Top-k results are sent with the question to the configured Groq chat model.
6. If the model omits a citation, the app adds one for the closest retrieved
   passage; citations to pages that were not retrieved are rejected.

The vector database is local under `data/chroma/`; it is excluded from Git.
The first run downloads the embedding model. Keep your API key private and do
not commit `.env`.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests
```
# ask-your-books
