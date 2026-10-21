# AI Coding Assistant with CRAG

A chat assistant that explains code, writes code with the help of a knowledge base of solved Python problems (HumanEval), checks if what it retrieved is really useful (Corrective RAG), searches Stack Overflow when its knowledge base has no answer, learns new solutions, runs code with a tool that the LLM calls, and remembers each conversation.

Built for the Cellula Technologies NLP internship: Project 2, part 1, with the Stack Overflow search added in part 2. The version graded for part 1 is kept at the [`part1` tag](https://github.com/HebaQasim/ai-coding-assistant-crag/tree/part1).

![The app](images/app.png)

## Features

- **RAG over HumanEval:** 164 solved Python problems are split with code chunking, embedded with `BAAI/bge-small-en-v1.5` and stored in Chroma.
- **Intent classifier and router:** every message is classified as `explain`, `generate` or `other`, and sent to the matching route.
- **Corrective RAG:** before writing code, an LLM checks if the retrieved problems really help. If not, the assistant does not guess: it searches Stack Overflow, and if that fails too, it asks the user for a correct solution.
- **Stack Overflow search with Tavily:** the assistant searches only stackoverflow.com, the LLM writes one clean solution from the best pages, the solution is checked, saved in Chroma with the page link, and used for the answer. The same request later finds it in Chroma, without a new search.
- **Human feedback learning:** a solution sent by the user is checked (syntax with `ast`, then an LLM review) and only then added to Chroma, with metadata.
- **Code runner tool:** the LLM is bound to a `run_code` tool and answers with a tool call. The code runs in a separate process with a time limit.
- **Memory:** conversations, messages and per-chat preferences are saved in SQLite. The LLM receives the preferences and the last 6 messages, so follow-up questions work.
- **Streamlit interface:** streaming answers, several chats, a view of what the LLM receives, file upload and a Run button.

## How a message flows

```mermaid
flowchart TD
    A["User message (app.py)"] --> B{"Waiting for a solution<br/>and the message is code?"}
    B -- yes --> L["learn.py: syntax check + LLM review"]
    L -- passed --> K["Saved in Chroma as user_feedback"]
    B -- no --> C["classifier.py: explain / generate / other"]
    C -- other --> O["Fixed reply, no LLM"]
    C -- explain --> E["explain.py: LLM explanation, no RAG"]
    C -- generate --> R["generate.py: rewrite follow-up questions"]
    R --> S["Chroma search, top 3"]
    S --> V{"relevance.py: relevant?"}
    V -- yes --> G["LLM writes the code"]
    V -- no --> W["web_search.py: Tavily search on Stack Overflow,<br/>the LLM writes a solution, it is checked"]
    W -- passed --> SO["Saved in Chroma as stack_overflow + link"]
    SO --> G
    W -- nothing found --> P["Ask the user for a solution"]
    G --> RUN["Run button: the LLM calls the run_code tool"]
```

Every answer and every run is saved in `memory.db`, and the memory is read again by the classifier, the relevance check and the LLM on the next message.

## Task parts and files

| Part | What it does | File |
|---|---|---|
| 1 | Explore the HumanEval dataset | `explore_data.py` |
| 2A | One Document per problem (prompt + solution), code chunking | `prepare_documents.py` |
| 2B | Embeddings and the Chroma vector database | `vector_db.py` |
| 3A | Gemini models through `ChatOpenAI`, with fallbacks | `llm.py`, `check_models.py` |
| 3B | Intent classifier, also picks up the user's preferences | `classifier.py` |
| 4A | Explanation route (no RAG) | `explain.py` |
| 4B | Router | `router.py` |
| 5A | Relevance check (the evaluator of CRAG) | `relevance.py` |
| 5B | Generation route: rewrite, retrieve, check, generate | `generate.py` |
| 6 | Human feedback learning | `learn.py` |
| 7 | Code runner tool and tool calling | `code_runner.py` |
| 8 | Conversation memory in SQLite | `memory.py` |
| 9 | Streamlit app | `app.py` |
| 10 | Stack Overflow search with Tavily (added in part 2) | `web_search.py` |

## Project structure

```text
.
├── app.py                 # Streamlit chat interface
├── explore_data.py        # look at the HumanEval data
├── prepare_documents.py   # Documents + code chunking
├── vector_db.py           # embeddings + Chroma
├── llm.py                 # Gemini models with fallbacks
├── check_models.py        # helper: which Gemini models can answer right now
├── classifier.py          # explain / generate / other + preferences
├── explain.py             # explanation route
├── router.py              # sends each message to its route
├── relevance.py           # relevance check (CRAG)
├── generate.py            # generation route
├── learn.py               # checks and saves new solutions (from the user or Stack Overflow)
├── web_search.py          # Stack Overflow search with Tavily
├── code_runner.py         # run_code tool + tool calling
├── memory.py              # SQLite memory: chats, messages, preferences
├── data/
│   └── humaneval.parquet  # HumanEval, downloaded from Hugging Face
├── requirements.txt
└── .env                   # your API keys (not in the repository)
```

`chroma_database/` and `memory.db` are created by the app when it runs.

## Setup

You need Python 3.10 or newer, a free Gemini API key from [Google AI Studio](https://aistudio.google.com/) and a free Tavily API key from [tavily.com](https://tavily.com/) (the free plan gives 1,000 searches a month).

```bash
git clone https://github.com/HebaQasim/ai-coding-assistant-crag.git
cd ai-coding-assistant-crag
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the packages and add your keys:

```bash
pip install -r requirements.txt
```

Create a file named `.env` in the project folder:

```text
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

Without a Tavily key the assistant still works: when the knowledge base has no answer, it asks the user for a solution.

Run the app:

```bash
streamlit run app.py
```

The first run downloads the embedding model (about 130 MB) and builds `chroma_database/` from HumanEval, so it takes a few minutes. Later runs open the saved database.

## Things to try

| Message | What happens |
|---|---|
| `I prefer short comments in my code` | Route `other`. The preference appears in the sidebar. |
| `Write a Python function that checks if a string is a palindrome` | Route `generate`, sources from HumanEval, relevance check ✅, a Run button under the code. |
| `Now make it ignore spaces and capital letters` | A follow-up: the request is rewritten for the search, the code is changed. |
| `Which line removes the spaces?` | Route `explain`, the answer uses the memory. |
| `Write a function that resizes an image with Pillow and saves the result` | Relevance check ❌, so the assistant searches Stack Overflow. The route is `web_search` and the source is `stack_overflow` with the page link. Ask again: the route is `generate` and the same source is found in Chroma. |
| A request when Stack Overflow has nothing useful (or without a Tavily key) | The assistant asks for a solution. Send one in a python code block: it is checked, then saved as `user_feedback`. |
| Click **Run code** | The LLM calls `run_code`. The output (or the error) is saved as a message, so you can ask "why did it fail?". |

## Testing each part

Each file has a small test that runs with `python <file>.py`. The number of Gemini requests is given because the free tier has a small daily limit.

| Command | What it checks | Gemini requests |
|---|---|---|
| `python explore_data.py` | shape, columns, one full problem, text lengths | 0 |
| `python prepare_documents.py` | 164 Documents, 164 chunks, metadata | 0 |
| `python vector_db.py` | number of chunks, top 3 results for 3 questions | 0 |
| `python llm.py` | one answer and the model that gave it | 1 |
| `python check_models.py` | each model on its own (useful after a 429 or 503 error) | 3 |
| `python code_runner.py` | normal code, error, infinite loop, `input()`, long output, tool call | 1 |
| `python learn.py` | not code, syntax error, wrong solution, correct solution | 2 |
| `python web_search.py` | the Stack Overflow search, what the LLM reads from each page, the solution and its check (saves nothing) | 2 + 1 Tavily search |
| `python generate.py` | the same new request twice: `web_search` the first time, `generate` the second time | 6 + 1 Tavily search |
| `python router.py` | a short conversation through all the routes (in a new chat) | about 15 |

Note: `python learn.py` saves its correct solution in Chroma each time it runs. `python generate.py` saves its solution the first time, so the next runs find it in Chroma.

## Design decisions

- **Code chunking:** the splitter cuts Python code between classes and functions first, so a function is not cut in half. With `chunk_size=2000`, every HumanEval problem (the longest has 1,993 characters) stays one chunk, with its description and its solution together.
- **Local embeddings:** `bge-small-en-v1.5` runs on the CPU, needs no API key and uses none of the Gemini quota.
- **An LLM as the relevance evaluator:** a distance score alone cannot tell "a similar problem with different words" from "a different problem with the same words". Any answer other than exactly `relevant` counts as not relevant, so code is never built on weak context.
- **Query rewriting:** a follow-up like "now write it in Java" is rewritten into a complete request before the search, using the conversation memory. The first message of a chat skips this step.
- **Two model chains:** `llm` starts with the strongest model for explanations and code; `fast_llm` starts with a light model for one-word decisions. Both use fallbacks, so a busy or exhausted model is replaced automatically.
- **Three checks before learning:** is it code, is the syntax valid, does an LLM review say it solves the request. The first two cost no request. A wrong solution in the knowledge base would mislead every later user, so anything unclear is rejected.
- **Stack Overflow as the corrective action:** like in the original CRAG paper, weak retrieval leads to a web search. The search is limited to stackoverflow.com, so the answers come from developers and are voted on.
- **The LLM writes the solution, it is not copied:** the pages hold pieces of several answers, and some code is incomplete or broken, so the LLM writes one complete function in the same shape as the HumanEval problems. It answers NONE when no page solves the request.
- **Reading a page from the question title:** the full page starts with the site menu and a long list of tags, so the text the LLM reads starts at the question title (10,000 characters: the question and the first answers). If the title is not found, the short Tavily snippets are used.
- **The same checks for every source:** a solution from Stack Overflow goes through the same syntax check and LLM review as a solution from the user. The review is done by another model, with its own prompt.
- **Code from the web is never run automatically:** it could need the network, passwords or real files, and the code runner is not a full sandbox. Running it stays the user's choice with the Run button.
- **The page link is saved:** the user can open the original answers, and Stack Overflow content (CC BY-SA) asks for attribution.
- **The code runner:** a separate process, an empty temporary folder, a 10 second limit, no keyboard input and at most 10,000 characters of output.
- **SQLite memory:** one file, part of Python, with transactions (a question is never saved without its answer) and per-chat preferences that the classifier replaces as a complete list.

## Limitations and future work

- **Free tier quota:** a `generate` message can use up to 4 Gemini requests (classifier, rewrite, relevance check, generation), and 6 when it searches Stack Overflow (plus writing and reviewing the solution). A paid plan removes this limit.
- **Web search results change:** the same request can return different pages, and sometimes the pages are only close to the request. The LLM then combines them, and the review decides if the solution can be saved.
- **The code runner is not a full sandbox:** the code can still read files or use the network. A real sandbox would run it in a Docker container or on a separate machine.
- **No duplicate check when learning:** the same solution can be saved twice. A future version could search Chroma first and skip very close chunks.
- **Tool calling is started by the Run button:** the next step is an agent loop where the LLM runs the code by itself, reads the error, fixes the code and runs it again.
- **The classifier can miss a preference** that is written in an unusual way, because it is an LLM decision.
- **One local user:** SQLite fits a single user on one computer. A shared deployment would use a server database such as PostgreSQL.
