import re
import uuid
from pathlib import Path
from config import settings

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt"
}


def is_supported_file(file_name: str) -> bool:
    """
    Check if the uploaded file type is supported.
    """
    extension = Path(file_name).suffix.lower()

    return extension in SUPPORTED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from filenames.
    """
    filename = Path(filename).stem
    filename = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        filename
    )
    filename = re.sub(
        r"\s+",
        "_",
        filename
    )

    return filename


def generate_chunk_id() -> str:
    """
    Generate a unique chunk ID.
    """
    return str(uuid.uuid4())


def clean_text(text: str) -> str:
    """
    Normalize extracted text.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def merge_metadata(*metadata):
    """
    Merge multiple metadata dictionaries.
    """
    merged = {}

    for item in metadata:
        if item:
            merged.update(item)

    return merged


def format_sources(chunks):
    """
    Convert retrieved chunks into
    a readable source list.
    """

    sources = []
    seen = set()

    for chunk in chunks:
        key = (
            chunk.source,
            chunk.page
        )
        if key in seen:
            continue
        seen.add(key)

        if chunk.page is not None:
            sources.append(
                f"{chunk.source} (Page {chunk.page})"
            )

        else:
            sources.append(chunk.source)

    return sources


def truncate_text(
    text,
    max_length=300
):
    """
    Truncate long strings for debugging.
    """

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def validate_chunk_size(chunks):
    """
    Simple validation after chunking.
    """

    valid_chunks = []

    for chunk in chunks:
        if len(chunk.text.strip()) < 20:
            continue
        valid_chunks.append(chunk)

    return valid_chunks


def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks.
    """

    unique = {}

    for chunk in chunks:
        unique[chunk.id] = chunk

    return list(unique.values())


def allowed_file_types():
    """
    Return Streamlit uploader file types.
    """

    return [
        "pdf",
        "docx",
        "ppt",
        "pptx",
        "txt"
    ]


def project_info():
    """
    Metadata shown in the app footer.
    """

    return {
        "name": "AI Study Assistant",
        "version": "1.0",
        "retrieval": "Hybrid RAG",
        "vector_store": "Qdrant",
        "graph": "Neo4j",
        "llm": settings.GROQ_MODEL
    }