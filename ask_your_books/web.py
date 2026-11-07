import re
import threading
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory, url_for
from openai import APIError

from .app import BookAssistant
from .config import Settings

_CITATION_PATTERN = re.compile(r"\[([^\]\n]+), p\. (\d+)\]")


def create_app(settings=None, assistant_factory=BookAssistant):
    settings = settings or Settings.from_environment()
    app = Flask(__name__)
    app.config["BOOKS_DIR"] = settings.books_dir
    app.config["ASSISTANT"] = None
    assistant_lock = threading.Lock()

    def get_assistant():
        if app.config["ASSISTANT"] is None:
            with assistant_lock:
                if app.config["ASSISTANT"] is None:
                    app.config["ASSISTANT"] = assistant_factory(settings)
        return app.config["ASSISTANT"]

    def render_room(answer=None, question="", message=None, message_kind="notice"):
        pdfs = sorted(settings.books_dir.rglob("*.pdf")) if settings.books_dir.is_dir() else []
        citations = []
        if answer:
            seen = set()
            for filename, page in _CITATION_PATTERN.findall(answer):
                citation = (filename, page)
                if citation not in seen:
                    seen.add(citation)
                    citations.append(
                        {
                            "filename": filename,
                            "page": page,
                            "url": url_for(
                                "book_pdf",
                                filename=filename,
                                _anchor="page={}".format(page),
                            ),
                        }
                    )
        return render_template(
            "reading_room.html",
            books=[
                {
                    "filename": pdf.relative_to(settings.books_dir).as_posix(),
                    "title": pdf.stem.replace("-", " ").replace("_", " "),
                }
                for pdf in pdfs
            ],
            answer=answer,
            citations=citations,
            question=question,
            message=message,
            message_kind=message_kind,
        )

    @app.get("/")
    def index():
        return render_room()

    @app.post("/ask")
    def ask():
        question = request.form.get("question", "").strip()
        if not question:
            return render_room(
                question=question,
                message="Write a question first, and I’ll look through the shelves.",
                message_kind="error",
            ), 400
        try:
            answer = get_assistant().answer(question)
        except (FileNotFoundError, RuntimeError, ValueError, APIError) as error:
            return render_room(
                question=question,
                message=str(error),
                message_kind="error",
            ), 400
        return render_room(answer=answer, question=question)

    @app.post("/ingest")
    def ingest():
        try:
            chunk_count = get_assistant().ingest()
        except (FileNotFoundError, RuntimeError, ValueError, APIError) as error:
            return render_room(
                message=str(error),
                message_kind="error",
            ), 400
        return render_room(
            message="The shelves are indexed — {} passages are ready to search.".format(
                chunk_count
            ),
        )

    @app.get("/books/<path:filename>")
    def book_pdf(filename):
        book_path = settings.books_dir / filename
        if not book_path.is_file() or book_path.suffix.lower() != ".pdf":
            abort(404)
        return send_from_directory(
            str(settings.books_dir.resolve()),
            filename,
            mimetype="application/pdf",
            as_attachment=False,
        )

    return app


def main():
    settings = Settings.from_environment()
    app = create_app(settings=settings)
    print("Ask-Your-Books is open at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
