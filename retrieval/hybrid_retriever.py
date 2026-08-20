from rank_bm25 import BM25Okapi
from models import Chunk, GraphRelation, RetrievalResult
from retrieval.vector_store import VectorStore
from graph.graph_retriever import GraphRetriever

class HybridRetriever:

    def __init__(self, vector_store,graph_retriever):

        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.bm25 = None
        self.corpus = []

    def build_keyword_index(self, chunks):

        self.corpus = chunks
        tokenized = [
            chunk.text.lower().split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query,
        semantic_k=5,
        keyword_k=5,
        final_k=10
    ):

        semantic_chunks = self.vector_store.similarity_search(
            query,
            semantic_k
        )

        keyword_chunks = self.keyword_search(
            query,
            keyword_k
        )

        primary_chunks = self.merge_chunks(
            semantic_chunks,
            keyword_chunks
        )

        graph_data = self.graph_retriever.retrieve(
            [chunk.id for chunk in primary_chunks]
        )

        graph_chunks = self.vector_store.get_chunks_by_ids(
            graph_data["related_chunk_ids"]
        )

        context_chunks = self.merge_chunks(
            primary_chunks,
            graph_chunks
        )

        context_chunks = self.rerank(
            query,
            context_chunks
        )

        context_chunks = context_chunks[:final_k]

        return RetrievalResult(
            context_chunks=context_chunks,
            primary_chunks=primary_chunks,
            graph_chunks=graph_chunks,
            related_concepts=graph_data["related_concepts"],
            relationships=[
                GraphRelation(**relation)
                for relation in graph_data["relationships"]
            ]
        )

    def keyword_search(
        self,
        query,
        k
    ):

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            zip(scores, self.corpus),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            chunk
            for score, chunk in ranked[:k]
            if score > 0
        ]

    def merge_chunks(
        self,
        first,
        second
    ):

        merged = {}

        for chunk in first + second:
            merged[chunk.id] = chunk

        return list(merged.values())

    def rerank(
        self,
        query,
        chunks
    ):

        query_embedding = (
            self.vector_store.embedding_model.embed_query(query)
        )

        scored = []

        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = (
                    self.vector_store.embedding_model.embed_documents(
                        [chunk.text]
                    )[0]
                )

            score = self.cosine_similarity(
                query_embedding,
                chunk.embedding
            )

            scored.append(
                (score, chunk)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            chunk
            for _, chunk in scored
        ]

    def cosine_similarity(
        self,
        a,
        b
    ):

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = (
            sum(x * x for x in a)
        ) ** 0.5

        norm_b = (
            sum(y * y for y in b)
        ) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot / (norm_a * norm_b)

    def clear(self):

        self.vector_store.clear()
        self.graph_retriever.clear()
        self.bm25 = None
        self.corpus = []

    def close(self):

        self.vector_store.close()
        self.graph_retriever.close()