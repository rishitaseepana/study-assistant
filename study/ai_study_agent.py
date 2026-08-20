import json

from langchain_core.prompts import ChatPromptTemplate

from study.study_tools import StudyTools


class AIStudyAgent:

    def __init__(self, retriever, llm):

        self.llm = llm
        self.tools = StudyTools(retriever)

        self.tool_map = {
            "study_plan": self.tools.generate_study_plan,
            "notes": self.tools.generate_notes,
            "explain": self.tools.explain_topic,
            "flashcards": self.tools.generate_flashcards,
            "quiz": self.tools.generate_quiz,
            "practice_questions": self.tools.generate_practice_questions
        }

        self.planner = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are an AI Study Agent.

Your task is to help students prepare for exams, interviews,
placements or revision.

You have access to the following tools:

study_plan
notes
explain
flashcards
quiz
practice_questions

Rules:

1. Understand the student's goal.
2. Select the minimum number of tools required.
3. Use at most four tools.
4. Return ONLY valid JSON.
5. Do not explain anything outside JSON.

Example:

Example:

{{
    "goal":"Prepare for DBMS interview",
    "steps":[
        {{
            "tool":"notes",
            "topic":"DBMS"
        }},
        {{
            "tool":"practice_questions",
            "topic":"DBMS",
            "count":10
        }},
        {{
            "tool":"quiz",
            "topic":"DBMS",
            "count":5
        }}
    ]
}}
"""
                    ),
                    (
                        "human",
                        "{goal}"
                    )
                ]
            )
        )

    def create_plan(self, goal):

        chain = self.planner | self.llm

        response = chain.invoke(
            {
                "goal": goal
            }
        )

        try:
            return json.loads(response.content)

        except Exception:

            raise ValueError(
                "Planner returned invalid JSON."
            )

    def execute_plan(self, plan):

        outputs = []

        for step in plan.get("steps", []):

            tool = step.get("tool")
            topic = step.get("topic")
            count = step.get("count", 10)

            if tool not in self.tool_map:

                outputs.append(
                    {
                        "tool": tool,
                        "error": "Unsupported tool."
                    }
                )

                continue

            function = self.tool_map[tool]

            try:

                if tool in ["flashcards", "quiz", "practice_questions"]:

                    result = function(
                        topic,
                        count
                    )

                else:

                    result = function(
                        topic
                    )

                outputs.append(
                    {
                        "tool": tool,
                        "topic": topic,
                        "result": result
                    }
                )

            except Exception as e:

                outputs.append(
                    {
                        "tool": tool,
                        "topic": topic,
                        "error": str(e)
                    }
                )

        return outputs

    def run(self, goal):

        plan = self.create_plan(goal)

        results = self.execute_plan(plan)

        return {
            "goal": plan.get("goal"),
            "plan": plan.get("steps", []),
            "results": results
        }