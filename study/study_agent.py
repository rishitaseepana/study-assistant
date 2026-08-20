from study.study_tools import StudyTools

class StudyAgent:

    def __init__(self,retriever):
        self.tools = StudyTools(retriever)

    def execute(
        self,
        action,
        topic,
        count=10
    ):

        action = action.lower().strip()
        actions = {
            "study_plan": lambda: self.tools.generate_study_plan(
                topic
            ),
            "notes": lambda: self.tools.generate_notes(
                topic
            ),
            "explain": lambda: self.tools.explain_topic(
                topic
            ),
            "flashcards": lambda: self.tools.generate_flashcards(
                topic,
                count
            ),
            "quiz": lambda: self.tools.generate_quiz(
                topic,
                count
            ),
            "practice_questions": lambda: (
                self.tools.generate_practice_questions(
                    topic,
                    count
                )
            )
        }

        if action not in actions:
            raise ValueError(
                f"Unsupported study action: {action}"
            )

        return actions[action]()

    def available_actions(self):
        return [
            "study_plan",
            "notes",
            "explain",
            "flashcards",
            "quiz",
            "practice_questions"
        ]