import argparse

from .app import BookAssistant


def main():
    parser = argparse.ArgumentParser(
        prog="ask-your-books",
        description="Ask questions about a folder of PDF books.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Index PDFs in a folder.")
    ingest_parser.add_argument("--pdf-dir", help="PDF folder (defaults to AYB_BOOKS_DIR).")

    ask_parser = commands.add_parser("ask", help="Ask one question about indexed PDFs.")
    ask_parser.add_argument("question", help="Question to answer.")

    args = parser.parse_args()
    assistant = BookAssistant()
    if args.command == "ingest":
        assistant.ingest(args.pdf_dir)
    else:
        print(assistant.answer(args.question))


if __name__ == "__main__":
    main()
