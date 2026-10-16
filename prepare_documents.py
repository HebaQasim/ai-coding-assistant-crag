"""Part 2 (step A): turn every HumanEval problem into a LangChain Document, then split the documents with code chunking."""
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def load_documents():
    """One Document per problem: the full function (prompt + solution) and its source in the metadata."""
    df = pd.read_parquet("data/humaneval.parquet")
    documents = []
    for _, row in df.iterrows():
        document = Document(
            page_content=row["prompt"] + row["canonical_solution"],
            metadata={"source": row["task_id"], "function_name": row["entry_point"]},
        )
        documents.append(document)
    return documents


def split_documents(documents):
    """Code chunking: Python code is split first between classes and functions, so a function is kept whole."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=2000,   # the longest function in HumanEval has 1,993 characters
        chunk_overlap=0,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))

    print("\nFirst chunk")
    print("Metadata:", chunks[0].metadata)
    print(chunks[0].page_content)
