import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from models import Chunk

class DocumentChunker:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                ""
            ]
        )

    def chunk_documents(self, documents):

        chunks = []
        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks

    def chunk_document(self, document):

        chunks = []

        for page_number, page_text in enumerate(document.pages, start=1):
            if not page_text.strip():
                continue
            split_chunks = self.splitter.split_text(page_text)

            for index, text in enumerate(split_chunks):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    page=page_number,
                    source=document.filename,
                    metadata={
                        "chunk_index": index,
                        "file_type": document.file_type,
                        "page": page_number,
                        "source": document.filename
                    }

                )
                chunks.append(chunk)

        return chunks