import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ask_your_books import app
from ask_your_books.core import (
    RetrievedChunk,
    UNKNOWN_ANSWER,
    add_citation_if_missing,
    build_prompt,
    parse_search_results,
    split_into_chunks,
    validate_citations,
)


class ChunkingTests(unittest.TestCase):
    def test_chunks_overlap_and_cover_words(self):
        words = ["word{}".format(index) for index in range(10)]
        chunks = split_into_chunks(" ".join(words), chunk_words=4, overlap_words=1)
        self.assertEqual(chunks, ["word0 word1 word2 word3", "word3 word4 word5 word6", "word6 word7 word8 word9"])

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_into_chunks("some text", chunk_words=4, overlap_words=4)


class RetrievalTests(unittest.TestCase):
    def test_parses_metadata_and_distance(self):
        chunks = parse_search_results(
            {
                "documents": [["A passage."]],
                "metadatas": [[{"filename": "book.pdf", "page": 3}]],
                "distances": [[0.2]],
            }
        )
        self.assertEqual(chunks[0].citation, "[book.pdf, p. 3]")
        self.assertEqual(chunks[0].distance, 0.2)

    def test_rejects_missing_page_metadata(self):
        with self.assertRaises(ValueError):
            parse_search_results(
                {
                    "documents": [["A passage."]],
                    "metadatas": [[{"filename": "book.pdf"}]],
                    "distances": [[0.2]],
                }
            )

    def test_prompt_includes_unknown_answer_and_citation_marker(self):
        source = RetrievedChunk("A passage.", "book.pdf", 3, 0.2)
        prompt = build_prompt("What happened?", [source])
        self.assertIn(UNKNOWN_ANSWER, prompt)
        self.assertIn("[book.pdf, p. 3]", prompt)

    def test_only_accepts_retrieved_citations(self):
        source = RetrievedChunk("A passage.", "book.pdf", 3, 0.2)
        self.assertEqual(
            validate_citations("The answer. [book.pdf, p. 3]", [source]),
            "The answer. [book.pdf, p. 3]",
        )
        with self.assertRaises(ValueError):
            validate_citations("The answer. [other.pdf, p. 9]", [source])
        with self.assertRaises(ValueError):
            validate_citations("An answer without a source.", [source])

    def test_adds_closest_retrieved_citation_when_model_omits_it(self):
        farther_source = RetrievedChunk("Other passage.", "other.pdf", 9, 0.4)
        closest_source = RetrievedChunk("Mars traits.", "astrology.pdf", 8, 0.2)

        answer = add_citation_if_missing(
            "Mars is associated with a strong, athletic build.",
            [farther_source, closest_source],
        )

        self.assertEqual(
            answer,
            "Mars is associated with a strong, athletic build. [astrology.pdf, p. 8]",
        )
        self.assertEqual(
            validate_citations(answer, [farther_source, closest_source]),
            answer,
        )

    def test_keeps_existing_retrieved_citation(self):
        source = RetrievedChunk("A passage.", "book.pdf", 3, 0.2)
        answer = "The answer. [book.pdf, p. 3]"

        self.assertEqual(add_citation_if_missing(answer, [source]), answer)


class GroqAnswerTests(unittest.TestCase):
    def setUp(self):
        self.assistant = object.__new__(app.BookAssistant)
        self.assistant.settings = SimpleNamespace(
            top_k=5,
            max_distance=0.55,
            groq_model="openai/gpt-oss-20b",
        )
        self.assistant.collection = SimpleNamespace(
            count=lambda: 1,
            query=lambda **kwargs: {
                "documents": [["Ariadne gave Theseus a thread."]],
                "metadatas": [[{"filename": "myths.pdf", "page": 12}]],
                "distances": [[0.2]],
            },
        )
        self.assistant.encoder = SimpleNamespace(
            encode=lambda *args, **kwargs: [
                SimpleNamespace(tolist=lambda: [0.1, 0.2])
            ]
        )

    def test_sends_question_to_groq_and_returns_cited_answer(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Ariadne gave Theseus a thread. [myths.pdf, p. 12]"
                    )
                )
            ]
        )
        completion = Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=completion)
            )
        )
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            with patch("ask_your_books.app.OpenAI", return_value=client) as groq_client:
                answer = self.assistant.answer("How did Ariadne help Theseus?")

        self.assertIn("Ariadne gave Theseus a thread.", answer)
        groq_client.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(
            completion.call_args.kwargs["model"],
            "openai/gpt-oss-20b",
        )

    def test_adds_retrieved_citation_when_groq_omits_it(self):
        self.assistant.collection.query = lambda **kwargs: {
            "documents": [["Mars has a strong, athletic build."]],
            "metadatas": [[
                {
                    "filename": "Astrology-How-to-Make-and-Read-Your-Own-Horoscope.pdf",
                    "page": 8,
                }
            ]],
            "distances": [[0.3]],
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Sepharial associates Mars with a strong, muscular, "
                            "athletic build and a ruddy complexion."
                        )
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=response))
            )
        )
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            with patch("ask_your_books.app.OpenAI", return_value=client):
                answer = self.assistant.answer(
                    "What physical traits does Sepharial associate with Mars?"
                )

        self.assertIn(
            "[Astrology-How-to-Make-and-Read-Your-Own-Horoscope.pdf, p. 8]",
            answer,
        )

    def test_out_of_corpus_question_does_not_require_groq_key(self):
        self.assistant.collection.query = lambda **kwargs: {
            "documents": [["An unrelated passage."]],
            "metadatas": [[{"filename": "myths.pdf", "page": 12}]],
            "distances": [[0.8]],
        }
        with patch.dict("os.environ", {}, clear=True):
            with patch("ask_your_books.app.OpenAI") as groq_client:
                answer = self.assistant.answer("What is the capital of Iceland?")

        self.assertEqual(answer, UNKNOWN_ANSWER)
        groq_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
