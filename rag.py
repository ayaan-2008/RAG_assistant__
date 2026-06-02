from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

from config import *

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

def ask(question):

    docs = db.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    sources = []

    for doc in docs:
        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        sources.append(
            f"{source} page {page}"
        )

    return {
        "answer": answer,
        "sources": list(set(sources))
    }