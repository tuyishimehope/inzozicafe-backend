from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUESTION_CHARS = 500  # every character is paid for twice: embedding and chat

SearchStrategy = Literal["semantic", "hybrid"]
AnswerStrategy = Literal[
    "semantic", "hybrid", "full"
]  # full = no retrieval, the token baseline


class Source(BaseModel):
    file: str
    element_id: str


class Hit(Source):
    text: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Answer(BaseModel):
    answer: str
    sources: list[Source] = []
    dropped_sources: list[Source] = []
    usage: Usage = Usage()


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    strategy: AnswerStrategy = "hybrid"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_name: str
    created_at: datetime
