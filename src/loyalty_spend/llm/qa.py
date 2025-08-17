"""Questions about the project answered from retrieved documentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from loyalty_spend.llm.client import LLMClient
from loyalty_spend.llm.corpus import Chunk
from loyalty_spend.llm.prompts import load_prompt
from loyalty_spend.llm.retrieval import TfidfRetriever
from loyalty_spend.llm.structured import complete_json

REFUSAL = "The documentation does not answer this question."


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Chunk]
    passages: list[Chunk] = field(repr=False, default_factory=list)

    @property
    def refused(self) -> bool:
        return self.answer.strip() == REFUSAL


def render_passages(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.citation})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def answer_question(
    client: LLMClient, retriever: TfidfRetriever, question: str, k: int = 4
) -> Answer:
    passages = [h.chunk for h in retriever.search(question, k)]
    if not passages:
        return Answer(question, REFUSAL, [], [])
    system, user = load_prompt("answer").render(
        passages=render_passages(passages), question=question
    )

    def valid_citations(data: dict) -> str | None:
        bad = [
            c
            for c in data["citations"]
            if not isinstance(c, int) or not 1 <= c <= len(passages)
        ]
        if bad:
            return f"citations {bad} do not refer to the numbered passages"
        if data["answer"].strip() != REFUSAL and not data["citations"]:
            return "an answer must cite at least one passage"
        return None

    reply = complete_json(
        client, system, user, {"answer": str, "citations": list}, valid_citations
    )
    cited = sorted(set(reply.data["citations"]))
    return Answer(
        question, reply.data["answer"], [passages[i - 1] for i in cited], passages
    )
