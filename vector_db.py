"""Part 2 (step B): turn the chunks into embeddings, store them in a Chroma vector database, and search it."""
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prepare_documents import load_documents, split_documents

DB_FOLDER = "chroma_database"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


def get_vector_db():
    """Open the saved database. If it does not exist yet, build it from the HumanEval chunks."""
    if os.path.exists(DB_FOLDER):
        return Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)

    chunks = split_documents(load_documents())
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_FOLDER,
    )


if __name__ == "__main__":
    vectordb = get_vector_db()
    print("Chunks in the database:", len(vectordb.get()["ids"]))

    questions = [
        "Write a function that checks if any two numbers in a list are closer than a given threshold",
        "Write a function that checks if a string is a palindrome",
        "Build a Flask API with a login endpoint",
    ]
    for question in questions:
        print("\nQuestion:", question)
        results = vectordb.similarity_search_with_score(question, k=3)
        for document, score in results:
            # solutions learned from the user have no function name, so .get() returns "-" for them
            print(f"  distance {score:.3f} | {document.metadata['source']} | {document.metadata.get('function_name', '-')}")
