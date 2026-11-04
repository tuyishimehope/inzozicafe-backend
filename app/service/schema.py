from pydantic import BaseModel


class Source(BaseModel):
    file: str
    element_id: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    dropped_sources: list[Source] = []
    usage: Usage = Usage()
