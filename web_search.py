"""Part 10: when the knowledge base cannot help, search Stack Overflow with Tavily, check the solution and learn it.
Step A: search_stack_overflow sends the request to Tavily and gets the best Stack Overflow pages.
Step B: the LLM writes one clean solution from those pages, the solution is checked, and only then saved in Chroma.

What happens when the relevance check says "not relevant":
    generate.py -> learn_from_web(request) -> Tavily search -> the LLM writes a solution -> check_solution
                -> learn_solution saves it in Chroma (source "stack_overflow" + the page link)
                -> generate.py answers the user with the new solution as the reference
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient

from code_runner import extract_code
from learn import check_solution, learn_solution
from llm import llm

load_dotenv()   # reads TAVILY_API_KEY from the .env file

MAX_RESULTS = 3          # the 3 best pages are enough, and they keep the prompt short
MAX_QUERY_LENGTH = 400   # Tavily does not accept longer queries
MAX_PAGE_LENGTH = 10000  # characters of each page, from the question title: the question and the first answers

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

SOLUTION_PROMPT = PromptTemplate.from_template(
    """You help an AI coding assistant learn a new solution from Stack Overflow.
The results below are the text of Stack Overflow pages: the question, then the answers, with words from the page
itself (like "Share" or "Copy link"). Some pages are cut at the end. Some code can be incomplete or wrong.

Request:
{request}

Stack Overflow results:
{results}

If the answers in the results show how to solve the request, write one complete and correct Python solution
based on them: the imports and a function with clear names and a docstring, inside one ```python block.
Do not add example calls. Before the code block, write one line: Result: <the number of the result you used most>
If no result shows how to solve the request (for example, the results only repeat the question), reply with NONE only."""
)


# ---------- Step A: search Stack Overflow ----------
def search_stack_overflow(request):
    """Search only stackoverflow.com and return the best pages."""
    # Each result is a dictionary, for example:
    #   {"title": "How to send an email with Python?",
    #    "url": "https://stackoverflow.com/questions/...",
    #    "content": "the parts of the page that look most like the request ...",
    #    "raw_content": "the text of the whole page: the question and all the answers ...",
    #    "score": 0.83}
    response = client.search(
        request[:MAX_QUERY_LENGTH],
        include_domains=["stackoverflow.com"],   # no blogs or other sites: only Stack Overflow answers
        max_results=MAX_RESULTS,
        include_raw_content="markdown",          # the whole page, so the answers are always there (same credit cost)
    )
    return response["results"]


def page_text(result):
    """The text of one page from the question title, cut to MAX_PAGE_LENGTH characters to keep the prompt short."""
    # The whole page starts with the site menu and a long list of tags; the question starts at its title.
    # If the title is not found (or Tavily could not read the page, so raw_content is None), we use the short content.
    text = result.get("raw_content") or ""
    start = text.find(result["title"])
    if start == -1:
        return result["content"]
    return text[start:start + MAX_PAGE_LENGTH]


def format_results(results):
    """Put the results in one numbered text for the prompt, so the LLM can say which one it used."""
    # Example:
    #   [1] How to send an email with Python?
    #   https://stackoverflow.com/questions/...
    #   the text of the page ...
    parts = []
    for number, result in enumerate(results, start=1):
        parts.append(f"[{number}] {result['title']}\n{result['url']}\n{page_text(result)}")
    return "\n\n".join(parts)


# ---------- Step B: write a solution, check it and learn it ----------
def write_solution(request, results):
    """The LLM writes one solution from the results. Return (code, url), or None if the results do not help."""
    answer = llm.invoke(SOLUTION_PROMPT.format(request=request, results=format_results(results))).content
    code = extract_code(answer)
    if code is None:   # the LLM replied NONE: the pages do not solve the request
        print("Stack Overflow: the results do not show a solution.")
        return None
    # The line "Result: 2" tells us which page the solution comes from. If it is missing or wrong, we use the first page.
    match = re.search(r"Result:\s*(\d+)", answer)
    number = int(match.group(1)) if match else 1
    if not 1 <= number <= len(results):
        number = 1
    return code, results[number - 1]["url"]


def learn_from_web(request):
    """Search -> write a solution -> check it -> save it. Return the saved Document, or None if nothing was learned."""
    try:
        results = search_stack_overflow(request)
    except Exception as error:   # no key, no internet or no credits left: the assistant will ask the user instead
        print("Tavily search failed:", error)
        return None
    if not results:
        print("Stack Overflow: no results.")
        return None

    solution = write_solution(request, results)
    if solution is None:
        return None
    code, url = solution

    problem = check_solution(request, code)   # the same checks as a solution from the user: syntax + LLM review
    if problem:
        print("The solution from Stack Overflow was not saved:", problem)
        return None
    return learn_solution(request, code, source="stack_overflow", url=url)


if __name__ == "__main__":
    # This test saves nothing in Chroma: it only shows each step (generate.py tests the saving)
    request = "Write a Python function that renames all the .txt files in a folder to .md"
    results = search_stack_overflow(request)
    print("===== Step A: what the LLM reads from each page (1 Tavily credit)")
    for number, result in enumerate(results, start=1):
        text = page_text(result)
        start = text[:100].replace("\n", " ")
        print(f"[{number}] {result['url']}")
        print(f"    {len(text):,} characters | {text.count('```') // 2} code blocks | starts with: {start}")

    print("\n===== Step B: the solution the LLM wrote (1 Gemini request)")
    solution = write_solution(request, results)
    if solution is not None:
        code, url = solution
        print("From:", url)
        print(code)
        print("===== The check (1 Gemini request):", check_solution(request, code) or "passed")
