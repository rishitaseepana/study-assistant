import time
from langchain_groq import ChatGroq
from config import settings

class RAGEvaluator:

    def __init__(self,llm):
        self.llm = llm

    def measure_latency(self, start_time):
        return round(time.time() - start_time, 3)

    def precision_at_k(
        self,
        retrieved_chunks,
        relevant_chunk_ids
    ):

        if not retrieved_chunks:
            return 0.0

        retrieved_ids = {
            chunk.id
            for chunk in retrieved_chunks
        }
        relevant_ids = set(relevant_chunk_ids)
        true_positive = len(
            retrieved_ids.intersection(relevant_ids)
        )
        return round(
            true_positive / len(retrieved_ids),
            3
        )

    def recall_at_k(
        self,
        retrieved_chunks,
        relevant_chunk_ids
    ):

        if not relevant_chunk_ids:
            return 0.0

        retrieved_ids = {
            chunk.id
            for chunk in retrieved_chunks
        }

        relevant_ids = set(relevant_chunk_ids)

        true_positive = len(
            retrieved_ids.intersection(relevant_ids)
        )

        return round(
            true_positive / len(relevant_ids),
            3
        )

    def context_relevance(
        self,
        question,
        chunks
    ):

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        prompt = f"""
            Rate how relevant the retrieved context is for answering the question.
            Question:
            {question}
            Context:
            {context}
            Reply with only one number between 0 and 1.
            """

        try:
            score = float(
                self.llm.invoke(prompt).content.strip()
            )

            return max(0.0, min(score, 1.0))

        except Exception:
            return 0.0

    def faithfulness(
        self,
        answer,
        chunks
    ):

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        prompt = f"""
            Determine whether the answer is fully supported by the context.
            Context:
            {context}
            Answer:
            {answer}
            Reply with only one number between 0 and 1.
            """

        try:
            score = float(
                self.llm.invoke(prompt).content.strip()
            )

            return max(0.0, min(score, 1.0))

        except Exception:

            return 0.0

    def answer_relevance(
        self,
        question,
        answer
    ):

        prompt = f"""
            Rate how well the answer addresses the user's question.
            Question:
            {question}
            Answer:
            {answer}
            Reply with only one number between 0 and 1.
            """

        try:

            score = float(
                self.llm.invoke(prompt).content.strip()
            )

            return max(0.0, min(score, 1.0))

        except Exception:

            return 0.0

    def evaluate(
        self,
        question,
        answer,
        retrieved_chunks,
        relevant_chunk_ids,
        latency
    ):

        return {

            "precision_at_k": self.precision_at_k(
                retrieved_chunks,
                relevant_chunk_ids
            ),
            "recall_at_k": self.recall_at_k(
                retrieved_chunks,
                relevant_chunk_ids
            ),
            "context_relevance": self.context_relevance(
                question,
                retrieved_chunks
            ),
            "faithfulness": self.faithfulness(
                answer,
                retrieved_chunks
            ),
            "answer_relevance": self.answer_relevance(
                question,
                answer
            ),
            "latency": latency
        }