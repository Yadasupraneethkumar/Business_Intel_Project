import sys

from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.chroma_vectorstore import ChromaVectorStore, Retriever


def search_vectorstore(
    query: str,
    top: int = 5,
    company: str | None = None,
    year: int | None = None
):
    """
    Search the local ChromaDB vector store and display
    the most relevant document chunks.
    """

    # Create the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to the existing ChromaDB collection
    store = ChromaVectorStore(
        collection_name="financial_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Create retriever
    retriever = Retriever(store.client)

    # Search ChromaDB
    results = retriever.invoke(
        query=query,
        company=company,
        year=year,
        top_k=top
    )

    print(f"Query: {query!r}")
    print(f"Top: {top}")

    if company:
        print(f"Company: {company}")

    if year:
        print(f"Year: {year}")

    print(f"Results: {len(results)}\n")

    # Display results
    for idx, document in enumerate(results, start=1):

        content = document.page_content

        snippet = (
            content.strip().replace("\n", " ")
            if isinstance(content, str)
            else "<no content>"
        )

        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  content snippet: {snippet}")

        if document.metadata:
            print(f"  metadata: {document.metadata}")

        print("  " + "-" * 60)

    if not results:
        print(
            "No results returned. "
            "Verify that ChromaDB contains documents "
            "or try a different query."
        )

    return results


def main():

    if len(sys.argv) < 2:
        print(
            'Usage: python -m rag.retrieval_debug '
            '"your query here"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    search_vectorstore(query)


if __name__ == "__main__":
    main()







