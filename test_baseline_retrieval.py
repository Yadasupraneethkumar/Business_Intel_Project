from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.weaviate_vectorstore import (
    WeaviateVectorStore,
    Retriever
)


def main():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = WeaviateVectorStore(
        collection_name="FinancialDocuments",
        embedding_function=embeddings
    )

    try:

        retriever = Retriever(vector_store)

        queries = [
            "What was Apple's total net sales revenue in 2024?",
            "What was Apple's net income in 2024?",
            "What was Apple's operating income in 2024?",
            "What were Apple's total assets in 2024?",
            "What were Apple's total liabilities in 2024?",
            "What was Apple's cash flow from operating activities in 2024?"
        ]

        for query in queries:

            print("\n" + "=" * 70)
            print(f"QUERY: {query}")
            print("=" * 70)

            results = retriever.invoke(
                query=query,
                company="Apple",
                year=2024,
                top_k=5
            )

            print(
                f"Retrieved: {len(results)} documents"
            )

            for index, document in enumerate(
                results,
                start=1
            ):
                print(
                    f"\n--- RESULT {index} ---"
                )

                print(
                    document.page_content[:500]
                )

                print(
                    "\nMetadata:",
                    document.metadata
                )

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()