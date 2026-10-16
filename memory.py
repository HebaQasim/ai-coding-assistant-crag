"""Part 8: long-term conversation memory, saved in a SQLite database (memory.db), so it is kept between sessions.
There can be many conversations (chats). Each chat has its own messages and its own preferences.

Two tables (one chat has many messages):
    conversations:  id | title (the first question) | preferences ("a; b") | created_at
    messages:       id | conversation_id | role (user/assistant) | text | route | sources ("HumanEval/48, ...") | created_at

Long-term memory: everything in memory.db.
Short-term memory: what the LLM receives with each request (as_text): the preferences + the last 6 messages of the chat.

Who uses what:
    new_conversation, list_conversations, delete_conversation, clear_preferences  ->  the sidebar of app.py
    add_exchange   ->  app.py after each answer and after each run, and the test in router.py
    set_preferences  ->  router.py, when the classifier finds new preferences
    last_question  ->  generate.py, to know if the chat already has a question before this one
    as_text  ->  the prompts of the classifier, explain, relevance, rewrite and generate steps
"""
import sqlite3

DB_FILE = "memory.db"
RECENT_MESSAGES = 6   # how many recent messages are sent to the LLM (3 questions and 3 answers)


class ConversationMemory:
    """Keeps the conversations: their questions and answers (with the code) and the user's preferences for each one."""

    def __init__(self, file=DB_FILE):
        # Opens memory.db, or creates an empty one if it does not exist yet.
        # check_same_thread=False: Streamlit can use the same connection from different threads
        self.connection = sqlite3.connect(file, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row   # read the columns of a row by name: row["text"] instead of row[1]
        # "with self.connection" is a transaction: everything inside is saved together, or nothing if there is an error.
        # CREATE TABLE IF NOT EXISTS only creates a missing table. It never changes an existing one,
        # so after a change in the columns the old memory.db must be deleted.
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT DEFAULT '',
                    preferences TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    route TEXT DEFAULT '',
                    sources TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
        # self.conversation_id is the chat that is open now: all the methods below work on this chat only
        self.open_latest_conversation()

    # Every value goes into the SQL with a "?" and is given separately in a tuple.
    # sqlite3 then treats it as plain data: a message like "What's wrong?" cannot break the SQL (no SQL injection).
    # A tuple with one value needs a comma: (self.conversation_id,)

    # ---------- Conversations ----------
    def create_conversation(self):
        # DEFAULT VALUES: a new row with an empty title, empty preferences and the current time
        with self.connection:
            cursor = self.connection.execute("INSERT INTO conversations DEFAULT VALUES")
        return cursor.lastrowid   # the id of the new chat, for example 3

    def open_latest_conversation(self):
        """Continue the most recent chat, or start the first one."""
        # The biggest id is the newest chat. fetchone() returns None when there is no chat yet (the first run).
        row = self.connection.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
        self.conversation_id = row["id"] if row else self.create_conversation()

    def new_conversation(self):
        """Start a new chat; the old chats stay saved. If the current chat is still empty, we simply keep it."""
        if self.messages:   # clicking "New chat" twice does not make two empty chats
            self.conversation_id = self.create_conversation()

    def list_conversations(self):
        """All the chats, newest first, with their titles (the first question of each chat)."""
        # Example: [{"id": 2, "title": "New chat"}, {"id": 1, "title": "Write a function that adds two numbers"}]
        rows = self.connection.execute("SELECT id, title FROM conversations ORDER BY id DESC").fetchall()
        return [{"id": row["id"], "title": row["title"] or "New chat"} for row in rows]

    def delete_conversation(self):
        """Delete the current chat and its messages, then open the most recent remaining chat."""
        # SQLite does not delete the messages of a chat by itself, so we delete them first, in the same transaction
        with self.connection:
            self.connection.execute("DELETE FROM messages WHERE conversation_id = ?", (self.conversation_id,))
            self.connection.execute("DELETE FROM conversations WHERE id = ?", (self.conversation_id,))
        self.open_latest_conversation()   # if no chat is left, a new empty one is created

    # ---------- Messages of the current chat ----------
    @property   # read like a variable: memory.messages (no brackets), always fresh from the database
    def messages(self):
        """All the messages of the current chat, oldest first."""
        # Example:
        #   [{"role": "user", "text": "Write a function that adds two numbers", "route": "generate", "sources": []},
        #    {"role": "assistant", "text": "def add(a, b): ...", "route": "generate", "sources": ["HumanEval/53"]}]
        rows = self.connection.execute(
            "SELECT role, text, route, sources FROM messages WHERE conversation_id = ? ORDER BY id",
            (self.conversation_id,),
        ).fetchall()
        return [
            {
                "role": row["role"],
                "text": row["text"],
                "route": row["route"],
                # saved as one text "HumanEval/48, HumanEval/10", returned as a list again
                "sources": row["sources"].split(", ") if row["sources"] else [],
            }
            for row in rows
        ]

    def add_exchange(self, question, answer, route="", sources=None):
        # Example: add_exchange("Write a function that adds two numbers", "def add(a, b): ...", "generate", ["HumanEval/53"])
        # adds two rows to messages: the user's question and the assistant's answer (with the sources).
        with self.connection:   # one transaction: a question is never saved without its answer
            self.connection.execute(
                "INSERT INTO messages (conversation_id, role, text, route) VALUES (?, ?, ?, ?)",
                (self.conversation_id, "user", question, route),
            )
            self.connection.execute(
                "INSERT INTO messages (conversation_id, role, text, route, sources) VALUES (?, ?, ?, ?, ?)",
                (self.conversation_id, "assistant", answer, route, ", ".join(sources or [])),
            )
            # The first question of a chat becomes its title in the list of chats:
            # its first line (a message can contain an uploaded file), at most 40 characters.
            # "AND title = ''" sets the title only once, so later questions do not change it.
            title = question.strip().splitlines()[0][:40]
            self.connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ''", (title, self.conversation_id)
            )

    def last_question(self):
        """The user's previous question in this chat ("" if the chat is new). Clicks on "Run code" are not questions."""
        # Example: "Write a function that adds two numbers", or "" in a new chat
        row = self.connection.execute(
            "SELECT text FROM messages WHERE conversation_id = ? AND role = 'user' AND route != 'run' "
            "ORDER BY id DESC LIMIT 1",
            (self.conversation_id,),
        ).fetchone()
        return row["text"] if row else ""

    # ---------- Preferences of the current chat ----------
    @property
    def preferences(self):
        """The preferences of this chat, as a list. They are saved as one text, separated by "; "."""
        # Example: "short comments; type hints" in the database  ->  ["short comments", "type hints"]
        row = self.connection.execute(
            "SELECT preferences FROM conversations WHERE id = ?", (self.conversation_id,)
        ).fetchone()
        return [p.strip() for p in row["preferences"].split(";") if p.strip()]

    def set_preferences(self, preferences):
        """Replace the preferences of this chat with the new complete list (old ones that changed are gone)."""
        # Example: set_preferences("detailed comments; type hints"). UPDATE writes over the old text, it does not add to it.
        with self.connection:
            self.connection.execute(
                "UPDATE conversations SET preferences = ? WHERE id = ?", (preferences, self.conversation_id)
            )

    def clear_preferences(self):
        self.set_preferences("")

    # ---------- What the LLM receives ----------
    def as_text(self):
        """The memory as text for the prompts: the preferences and the recent messages of this chat."""
        # Example of the result:
        #   User preferences:
        #   - short comments
        #   - type hints
        #
        #   Recent conversation:
        #   USER: Write a function that adds two numbers
        #
        #   ASSISTANT: def add(a, b): ...
        preferences = "\n".join(f"- {p}" for p in self.preferences) or "None yet"
        # To get the LAST 6 messages we sort newest first and take 6 ...
        rows = self.connection.execute(
            "SELECT role, text FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (self.conversation_id, RECENT_MESSAGES),
        ).fetchall()
        recent = reversed(rows)   # ... then reverse them: the prompt needs the conversation in its real order
        conversation = "\n\n".join(f"{row['role'].upper()}: {row['text']}" for row in recent) or "No previous messages"
        return f"User preferences:\n{preferences}\n\nRecent conversation:\n{conversation}"
