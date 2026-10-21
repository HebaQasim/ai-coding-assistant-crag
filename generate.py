"""Part 5 (step B): the code generation route: rewrite the request -> retrieve from Chroma -> relevance check -> generate code.
Part 10: if the retrieved code is not relevant, search Stack Overflow first, and ask the user only if that fails too."""
from langchain_core.prompts import PromptTemplate

from llm import fast_llm, llm
from relevance import format_context, is_relevant, source_of
from vector_db import get_vector_db
from web_search import learn_from_web

TOP_K = 3

REWRITE_PROMPT = PromptTemplate.from_template(
    """You prepare the search query of an AI coding assistant.
The assistant will search its knowledge base of Python functions with this query.
If the new message is a follow-up that only makes sense with the conversation (for example "now write it in Java"),
rewrite it as one complete request that names the task, for example:
"Write a Java function that checks if a string is a palindrome".
If the new message is already complete on its own, or starts a new topic, return it unchanged.
Reply with the request only.

Conversation memory:
{memory}

New message:
{message}"""
)

NOT_RELEVANT_MESSAGE = (
    "I couldn't find relevant knowledge for your request, in my knowledge base or on Stack Overflow. "
    "Could you provide the correct solution or reference so I can learn it for future interactions?"
)

GENERATE_PROMPT = PromptTemplate.from_template(
    """You are an expert developer inside an AI coding assistant.
Write code for the user's request, in Python unless the request or the user's preferences ask for another language.
Use the reference code below as a guide: it comes from the assistant's knowledge base and solves a similar problem.
Follow the user's preferences from the conversation memory, and use the recent conversation for follow-up requests
(for example "now make it faster" means: change the code written before).

Your answer must contain:
- The complete source code in one Markdown code block (```python)
- Clear comments
- Best practices: clear names, type hints and a docstring
- Error handling for invalid input
- At the end of the code, a short example that calls the code and prints the result (do not use input())
- After the code block, a short explanation (2-3 sentences) that mentions which reference source helped

Conversation memory:
{memory}

Reference code:
{context}

User request:
{message}"""
)

vectordb = get_vector_db()


def make_search_query(message, memory):
    """Turn a follow-up like "now write it in Java" into a complete request, so the search finds the right topic."""
    if not memory.last_question():   # the first request of a chat has nothing to depend on: no LLM call needed
        return message
    prompt = REWRITE_PROMPT.format(memory=memory.as_text(), message=message)
    return fast_llm.invoke(prompt).content.strip() or message


def generate_code(message, memory):
    """Return the route ("generate" or "web_search"), the sources and the answer as a stream of text pieces."""
    query = make_search_query(message, memory)
    documents = vectordb.similarity_search(query, k=TOP_K)
    route = "generate"

    if not is_relevant(query, documents, memory):
        # Corrective action: search Stack Overflow. If a checked solution is found, it is saved in Chroma
        # and it becomes the reference for this answer.
        learned = learn_from_web(query)
        if learned is None:   # nothing found, or the solution did not pass the checks: ask the user
            sources = [source_of(document) for document in documents]
            return route, sources, [NOT_RELEVANT_MESSAGE]   # a list with one piece, so the caller can treat it like a stream
        documents = [learned]
        route = "web_search"

    sources = [source_of(document) for document in documents]
    prompt = GENERATE_PROMPT.format(memory=memory.as_text(), context=format_context(documents), message=message)
    return route, sources, (chunk.content for chunk in llm.stream(prompt))


if __name__ == "__main__":
    from memory import ConversationMemory

    memory = ConversationMemory()
    memory.new_conversation()   # the test runs in a new chat; the old chats stay saved

    # HumanEval has nothing about files: the first time, the assistant searches Stack Overflow and learns a solution.
    # The second time, the same request finds the saved solution in Chroma, so there is no web search.
    request = "Write a Python function that renames all the .txt files in a folder to .md"
    for attempt in [1, 2]:
        route, sources, stream = generate_code(request, memory)
        print(f"===== Attempt {attempt}")
        print("Route:", route)
        print("Sources:", sources)
        print("".join(stream))
