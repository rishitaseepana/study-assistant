import re
from langchain_groq import ChatGroq
from config import settings
import json

class GuardRails:

    def __init__(self):

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0
        )

        self.injection_patterns = [
            r"ignore previous",
            r"ignore all instructions",
            r"system prompt",
            r"developer message",
            r"reveal prompt",
            r"jailbreak",
            r"bypass",
            r"override",
            r"forget previous",
            r"act as",
            r"pretend to be"
        ]

    def validate_question(self, question):

        question = question.strip()

        if len(question) == 0:
            return False, "Please enter a question."

        if len(question) > 2000:
            return False, "Question is too long."

        for pattern in self.injection_patterns:
            if re.search(
                pattern,
                question,
                flags=re.IGNORECASE
            ):

                return (
                    False,
                    "Potential prompt injection detected."
                )

        return True, ""

    def validate_context(self, chunks):

        cleaned = []
        seen = set()

        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            if text in seen:
                continue

            seen.add(text)
            cleaned.append(chunk)

        if len(cleaned) == 0:
            return (
                False,
                [],
                "No relevant information was found."
            )

        return (
            True,
            cleaned,
            ""
        )

    def validate_answer(
        self,
        question,
        answer,
        chunks
    ):

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        prompt = f"""
            You are checking whether an answer is fully supported by the provided context.
            Question:
            {question}
            Context:
            {context}
            Answer:
            {answer}
            Reply ONLY with JSON.
            {{
                "grounded": true,
                "reason": ""
            }}
            """

        try:
            response = self.llm.invoke(
                prompt
            ).content
            response = json.loads(response)
            if response["grounded"]:
                return True, ""

            return (
                False,
                response["reason"]
            )

        except Exception:
            return (
                True,
                ""
            )

    def filter_sources(self, chunks):

        unique = {}

        for chunk in chunks:
            key = (
                chunk.source,
                chunk.page
            )

            if key not in unique:
                unique[key] = chunk

        return list(unique.values())