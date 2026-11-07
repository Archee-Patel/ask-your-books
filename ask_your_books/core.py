import re
from dataclasses import dataclass


UNKNOWN_ANSWER = "I don't know based on the provided documents."
_CITATION_PATTERN = re.compile(r"\[([^\]\n]+), p\. (\d+)\]")


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int
    distance: float

    @property
    def citation(self):
        return "[{}, p. {}]".format(self.filename, self.page)


def split_into_chunks(text, chunk_words=220, overlap_words=40):
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def validate_citations(answer, sources):
    allowed = {(source.filename, str(source.page)) for source in sources}
    citations = _CITATION_PATTERN.findall(answer)
    if not citations:
        raise ValueError("The LLM answer did not include a valid document citation.")
    invalid = [(filename, page) for filename, page in citations if (filename, page) not in allowed]
    if invalid:
        raise ValueError(
            "The LLM answer cited a page that was not retrieved: {}.".format(invalid[0])
        )
    return answer


def add_citation_if_missing(answer, sources):
    if _CITATION_PATTERN.search(answer):
        return answer
    if not sources:
        raise ValueError("Cannot cite an answer without retrieved document passages.")
    closest_source = min(sources, key=lambda source: source.distance)
    return "{} {}".format(answer.rstrip(), closest_source.citation)


def build_prompt(question, sources):
    context = "\n\n".join(
        "{}\n{}".format(source.citation, source.text) for source in sources
    )
    return (
        "Answer the question using only the document excerpts below. "
        "End every answer with at least one exact source marker shown before an excerpt, "
        "using the format [filename, p. N]. Cite every factual claim. Do not invent "
        "citations or facts. "
        "If the excerpts do not contain enough information, reply exactly: "
        "\"{}\"\n\n"
        "Document excerpts:\n{}\n\nQuestion: {}"
    ).format(UNKNOWN_ANSWER, context, question)


def parse_search_results(results):
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    if not (len(documents) == len(metadatas) == len(distances)):
        raise ValueError("Vector search returned incomplete result metadata.")
    chunks = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if not document or not metadata:
            continue
        try:
            filename = str(metadata["filename"])
            page = int(metadata["page"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("A retrieved chunk is missing its filename or page.") from error
        chunks.append(
            RetrievedChunk(
                text=document,
                filename=filename,
                page=page,
                distance=float(distance),
            )
        )
    return chunks
