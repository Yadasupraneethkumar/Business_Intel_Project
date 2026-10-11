
import sys

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.weaviate_vectorstore import (
    WeaviateVectorStore,
    Retriever,
)

load_dotenv()


def search_vectorstore(
    query: str,
    top: int = 5,
    company: str | None = None,
    year: int | None = None,
):
    """Search Weaviate Cloud and display relevant document chunks."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = None

    try:
        store = WeaviateVectorStore(
            collection_name="FinancialDocuments",
            embedding_function=embeddings,
        )

        retriever = Retriever(store)

        results = retriever.invoke(
            query=query,
            company=company,
            year=year,
            top_k=top,
        )

        print(f"Query: {query!r}")
        print(f"Top: {top}")
        print(f"Company: {company or 'All'}")
        print(f"Year: {year or 'All'}")
        print(f"Results: {len(results)}\n")

        for idx, document in enumerate(results, start=1):
            snippet = document.page_content.strip().replace("\n", " ")
            if len(snippet) > 350:
                snippet = snippet[:350].rstrip() + "..."

            print(f"Result {idx}")
            print(f"  Content: {snippet}")
            print(f"  Metadata: {document.metadata}")
            print("-" * 60)

        if not results:
            print("No results returned from Weaviate.")

        return results

    finally:
        if store is not None:
            store.close()


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: uv run python -m rag.retrieval_debug '
            '"your query here"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()