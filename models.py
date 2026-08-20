from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    filename: str
    file_type: str
    text: str
    pages: List[str]


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    source: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class GraphRelation:
    source: str
    relationship: str
    target: str


@dataclass
class RetrievalResult:
    context_chunks: List[Chunk]
    primary_chunks: List[Chunk]
    graph_chunks: List[Chunk]
    related_concepts: List[str]
    relationships: List[GraphRelation]


@dataclass
class ChatResponse:
    answer: str
    sources: List[Chunk]
    related_concepts: List[str]
    relationships: List[GraphRelation]


@dataclass
class FlashCard:
    front: str
    back: str


@dataclass
class QuizQuestion:
    question: str
    answer: str
    difficulty: str = "Medium"


@dataclass
class StudyPlan:
    title: str
    schedule: List[str]


@dataclass
class StudyMaterial:
    title: str
    content: str