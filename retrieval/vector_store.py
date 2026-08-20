from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct
)
from config import settings
from models import Chunk
from retrieval.embeddings import EmbeddingModel
from pathlib import Path

class VectorStore:

    def __init__(self, workspace_path):

        self.embedding_model = EmbeddingModel()
        self.qdrant_path = Path(workspace_path) / "qdrant"
        self.qdrant_path.mkdir(
            parents=True,
            exist_ok=True
        )
        self.client = QdrantClient(
            path=str(self.qdrant_path)
        )
        self.collection = settings.QDRANT_COLLECTION
        self._create_collection()

    def _create_collection(self):

        collections = [
            c.name
            for c in self.client.get_collections().collections
        ]

        if self.collection in collections:
            return

        dimension = len(
            self.embedding_model.embed_query("dimension")
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )

    def add_chunks(self, chunks):

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "page": chunk.page,
                        "source": chunk.source,
                        "metadata": chunk.metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            wait=True,
            points=points
        )

    def similarity_search(
        self,
        query,
        k=5
    ):

        query_embedding = self.embedding_model.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=k
        ).points

        return [
            self._point_to_chunk(point)
            for point in results
        ]

    def get_chunks_by_ids(self, ids):

        if not ids:
            return []

        results = self.client.retrieve(
            collection_name=self.collection,
            ids=ids
        )

        return [
            self._point_to_chunk(point)
            for point in results
        ]

    def _point_to_chunk(self, point):

        payload = point.payload

        return Chunk(
            id=str(point.id),
            text=payload["text"],
            page=payload["page"],
            source=payload["source"],
            metadata=payload.get("metadata", {}),
            embedding=point.vector
        )

    def count(self):

        return self.client.count(
            collection_name=self.collection
        ).count
    
    def is_empty(self):

        return self.count() == 0

    def clear(self):

        if not self.is_empty():

            collections = [
                c.name
                for c in self.client.get_collections().collections
            ]

            if self.collection in collections:
                self.client.delete_collection(
                    self.collection
                )

            self._create_collection()

    def close(self):
        self.client.close()