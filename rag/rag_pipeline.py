import json
from langchain_groq import ChatGroq
from config import settings
from models import ChatResponse
from rag.guardrails import GuardRails
from retrieval.hybrid_retriever import HybridRetriever
from graph.graph_builder import GraphBuilder
from graph.graph_retriever import GraphRetriever
from retrieval.vector_store import VectorStore
from evaluation.evaluation import RAGEvaluator
from study.study_tools import StudyTools

class RAGPipeline:

    def __init__(self, workspace_path, workspace_name):

        self.workspace_path = workspace_path
        self.vector_store = VectorStore(workspace_path)
        self.graph_builder = GraphBuilder(workspace_name)
        self.graph_retriever = GraphRetriever(workspace_name)
        self.retriever = HybridRetriever(
            self.vector_store,
            self.graph_retriever
        )
        self.guardrails = GuardRails()
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.TEMPERATURE
        )
        self.chat_history = []
        self.evaluator = RAGEvaluator(self.llm)
        self.study_tools = StudyTools(self.retriever)

    def ask(
        self,
        question,
        debug=False
    ):

        valid, message = self.guardrails.validate_question(
            question
        )

        if not valid:
            raise ValueError(message)

        retrieval = self.retriever.retrieve(question)
        valid, chunks, message = (
            self.guardrails.validate_context(
                retrieval.context_chunks
            )
        )

        if not valid:
            raise ValueError(message)

        context = self.build_context(chunks)
        prompt = self.build_prompt(
            question,
            context
        )

        answer = self.llm.invoke(prompt).content.strip()
        grounded, reason = (
            self.guardrails.validate_answer(
                question,
                answer,
                chunks
            )
        )

        if not grounded:
            answer = (
                "I couldn't answer this confidently using the uploaded documents.\n\n"
                f"Reason: {reason}"
            )

        self.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        response = ChatResponse(
            answer=answer,
            sources=self.guardrails.filter_sources(
                chunks
            ),
            related_concepts=retrieval.related_concepts,
            relationships=retrieval.relationships
        )

        if debug:
            return {
                "response": response,
                "debug": {
                    "primary_chunks": retrieval.primary_chunks,
                    "graph_chunks": retrieval.graph_chunks,
                    "context_chunks": retrieval.context_chunks,
                    "related_concepts": retrieval.related_concepts,
                    "relationships": retrieval.relationships
                }
            }

        return response

    def build_context(
        self,
        chunks
    ):

        context = []

        for chunk in chunks:
            context.append(
                f"""Source: {chunk.source}
                    Page: {chunk.page}
                    {chunk.text}
                    """
            )

        return "\n\n----------------------\n\n".join(
            context
        )

    def build_prompt(
        self,
        question,
        context
    ):

        history = ""

        if self.chat_history:
            recent = self.chat_history[-5:]
            for chat in recent:

                history += (
                    f"User: {chat['question']}\n"
                    f"Assistant: {chat['answer']}\n\n"
                )

        return f"""
            You are an AI Study Assistant.
            Answer ONLY using the provided context.
            If the answer is not present in the context,
            say you do not know.
            Keep the answer clear and suitable for students.
            Conversation History:
            {history}
            Context:
            {context}
            Question:
            {question}
            """

    def clear_chat(self):
        self.chat_history = []

    def load_chat(self, history):
        self.chat_history = history

    def get_chat(self):
        return self.chat_history

    def ingest_documents(
        self,
        chunks
    ):
        
        self.graph_builder.clear()
        self.vector_store.add_chunks(
            chunks
        )
        self.graph_builder.build(
            chunks
        )
        self.retriever.build_keyword_index(
            chunks
        )

    def export_chat(self):
        return self.chat_history

    def import_chat(self, history):
        self.chat_history = history

    def close(self):
        self.retriever.close()