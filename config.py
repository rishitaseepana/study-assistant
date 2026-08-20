import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION",
        "study_assistant"
    )

    NEO4J_URI: str = os.getenv(
        "NEO4J_URI",
        "bolt://neo4j:7687"
    )
    NEO4J_USERNAME: str = os.getenv(
        "NEO4J_USERNAME",
        "neo4j"
    )
    NEO4J_PASSWORD: str = os.getenv(
        "NEO4J_PASSWORD",
        "password"
    )

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))

    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 5))
    GRAPH_HOPS: int = int(os.getenv("GRAPH_HOPS", 2))

    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.2))

    ALLOWED_FILE_TYPES: tuple = (
        ".pdf",
        ".docx",
        ".ppt",
        ".pptx",
        ".txt"
    )

settings = Settings()