"""Part 5 (step A): the relevance check. Another LLM call decides if the retrieved code really helps with the request."""
from langchain_core.prompts import PromptTemplate

from llm import fast_llm

RELEVANCE_PROMPT = PromptTemplate.from_template(
    """You are the relevance evaluator of an AI coding assistant.
The assistant searched its knowledge base of Python functions to help with the user's request.
Decide if the retrieved code is relevant enough to help answer the request.
It is relevant if it solves the same problem or a similar one, so it can guide the answer
(small differences are fine: the assistant will adapt the code).
It is not relevant if it is about a different problem and only shares some words with the request.
If the request is a follow-up that changes earlier code (for example "now make it ignore spaces"),
use the conversation memory and judge the retrieved code against the topic of the conversation (for example palindromes).

Conversation memory:
{memory}

User request:
{message}

Retrieved code:
{context}

Reply with one word only: relevant or not_relevant."""
)


def format_context(documents):
    """Put the retrieved chunks in one text, each one under its source."""
    parts = []
    for document in documents:
        parts.append(f"# Source: {document.metadata['source']}\n{document.page_content}")
    return "\n\n".join(parts)


def is_relevant(message, documents, memory):
    """Return True only if the LLM answers exactly "relevant"."""
    prompt = RELEVANCE_PROMPT.format(memory=memory.as_text(), message=message, context=format_context(documents))
    answer = fast_llm.invoke(prompt).content.strip().lower().rstrip(".")
    return answer == "relevant"   # any other answer counts as not relevant, so we never build on weak context
