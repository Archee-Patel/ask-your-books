import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    books_dir: Path
    chroma_dir: Path
    embedding_model: str
    groq_model: str
    top_k: int
    max_distance: float

    @classmethod
    def from_environment(cls):
        load_dotenv()
        top_k = int(os.getenv("AYB_TOP_K", "5"))
        max_distance = float(os.getenv("AYB_MAX_DISTANCE", "0.55"))
        if top_k < 1:
            raise ValueError("AYB_TOP_K must be at least 1.")
        if not 0 <= max_distance <= 2:
            raise ValueError("AYB_MAX_DISTANCE must be between 0 and 2.")
        return cls(
            books_dir=Path(os.getenv("AYB_BOOKS_DIR", "data/books")),
            chroma_dir=Path(os.getenv("AYB_CHROMA_DIR", "data/chroma")),
            embedding_model=os.getenv(
                "AYB_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            top_k=top_k,
            max_distance=max_distance,
        )
