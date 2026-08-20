from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

class EmbeddingModel:

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, query):
        return self.model.embed_query(query)