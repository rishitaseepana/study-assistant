from langchain_groq import ChatGroq
from config import settings
from retrieval.hybrid_retriever import HybridRetriever

class StudyTools:

    def __init__(self, retriever):

        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.3
        )

    def generate_study_plan(self, topic):

        context = self._get_context(topic)

        prompt = f"""
            You are an AI Study Assistant.
            Using ONLY the provided context, create a structured study plan.
            Include:
            - Learning objectives
            - Topics to study
            - Recommended order
            - Revision tips
            - Practice suggestions
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def generate_notes(self, topic):

        context = self._get_context(topic)

        prompt = f"""
            Using ONLY the provided context, create concise study notes.
            Requirements:
            - Headings
            - Bullet points
            - Key definitions
            - Important concepts
            - Summary
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def explain_topic(self, topic):

        context = self._get_context(topic)

        prompt = f"""
            Explain the topic using ONLY the provided context.
            The explanation should be:
            - Easy to understand
            - Well structured
            - Student friendly
            - Include examples if available in the context
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def generate_flashcards(
        self,
        topic,
        count=10
    ):

        context = self._get_context(topic)

        prompt = f"""
            Using ONLY the provided context, generate {count} flashcards.
            Format exactly as:
            Q:
            A:
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def generate_quiz(
        self,
        topic,
        count=10
    ):

        context = self._get_context(topic)

        prompt = f"""
            Create {count} multiple-choice questions using ONLY the provided context.
            Each question should contain:
            - Question
            - Four options
            - Correct answer
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def generate_practice_questions(
        self,
        topic,
        count=10
    ):

        context = self._get_context(topic)

        prompt = f"""
            Generate {count} descriptive practice questions using ONLY the provided context.
            Do NOT provide solutions.
            Context:
            {context}
            Topic:
            {topic}
            """

        return self.llm.invoke(prompt).content

    def generate_answers(
        self,
        questions
    ):

        prompt = f"""
            Provide detailed answers for the following questions.
            Use ONLY the uploaded documents.
            Questions:
            {questions}
            """

        return self.llm.invoke(prompt).content

    def _get_context(self, topic):

        retrieval = self.retriever.retrieve(topic)
        context = []

        for chunk in retrieval.context_chunks:
            context.append(chunk.text)

        return "\n\n".join(context)