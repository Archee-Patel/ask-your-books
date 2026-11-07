import unittest
from pathlib import Path
from unittest.mock import Mock

from ask_your_books.config import Settings
from ask_your_books.web import create_app


ROOT = Path(__file__).resolve().parents[1]


class ReadingRoomTests(unittest.TestCase):
    def setUp(self):
        self.assistant = Mock()
        self.settings = Settings(
            books_dir=ROOT / "data" / "books",
            chroma_dir=ROOT / "data" / "chroma",
            embedding_model="unused-in-web-tests",
            groq_model="unused-in-web-tests",
            top_k=5,
            max_distance=0.55,
        )
        self.app = create_app(
            settings=self.settings,
            assistant_factory=lambda settings: self.assistant,
        )
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_homepage_lists_books_without_loading_model(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The little", response.data)
        self.assertIn(b"Bulfinch s Mythology", response.data)
        self.assertIsNone(self.app.config["ASSISTANT"])

    def test_question_shows_answer_and_clickable_page_citation(self):
        self.assistant.answer.return_value = (
            "Ariadne helped Theseus. [Bulfinch-s-Mythology.pdf, p. 78]"
        )

        response = self.client.post(
            "/ask",
            data={"question": "Who helped Theseus?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ariadne helped Theseus.", response.data)
        self.assertIn(b"/books/Bulfinch-s-Mythology.pdf#page=78", response.data)
        self.assistant.answer.assert_called_once_with("Who helped Theseus?")

    def test_empty_question_is_reported(self):
        response = self.client.post("/ask", data={"question": " "})

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Write a question first", response.data)
        self.assistant.assert_not_called()

    def test_index_button_runs_ingestion(self):
        self.assistant.ingest.return_value = 120

        response = self.client.post("/ingest")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"120 passages are ready to search", response.data)
        self.assistant.ingest.assert_called_once_with()

    def test_pdf_route_serves_book(self):
        response = self.client.get("/books/Bulfinch-s-Mythology.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF"))
        response.close()

    def test_pdf_route_rejects_missing_book(self):
        response = self.client.get("/books/not-a-book.pdf")

        self.assertEqual(response.status_code, 404)

    def test_pdf_route_resolves_relative_books_directory(self):
        relative_settings = Settings(
            books_dir=Path("data/books"),
            chroma_dir=self.settings.chroma_dir,
            embedding_model=self.settings.embedding_model,
            groq_model=self.settings.groq_model,
            top_k=self.settings.top_k,
            max_distance=self.settings.max_distance,
        )
        app = create_app(
            settings=relative_settings,
            assistant_factory=lambda settings: self.assistant,
        )
        app.config["TESTING"] = True

        with app.test_client() as client:
            response = client.get("/books/Bulfinch-s-Mythology.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"%PDF"))
        response.close()


if __name__ == "__main__":
    unittest.main()
