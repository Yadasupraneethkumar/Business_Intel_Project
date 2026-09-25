from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.weaviate_vectorstore import (
    WeaviateVectorStore,
    Retriever
)


def main():

    print("=" * 70)
    print("STEP 1: Loading embedding model")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded successfully.\n")


    print("=" * 70)
    print("STEP 2: Connecting to Weaviate")
    print("=" * 70)

    vector_store = WeaviateVectorStore(
        collection_name="FinancialDocuments",
        embedding_function=embeddings
    )

    print("Weaviate vector store initialized successfully.\n")


    print("=" * 70)
    print("STEP 3: Creating test document")
    print("=" * 70)

    test_document = type(
        "TestDocument",
        (),
        {
            "page_content": (
                "Apple reported strong revenue growth "
                "during the fiscal year."
            ),
            "metadata": {}
        }
    )()

    print("Test document created.\n")


    print("=" * 70)
    print("STEP 4: Uploading test document")
    print("=" * 70)

    vector_store.upload_chunks(
        chunks=[test_document],
        embeddings=embeddings,
        company="Apple",
        year="2024",
        source_file="test_2024_Apple.pdf"
    )

    print("Test document uploaded successfully.\n")


    print("=" * 70)
    print("STEP 5: Testing retrieval")
    print("=" * 70)

    retriever = Retriever(vector_store)

    results = retriever.invoke(
        query="What happened to Apple's revenue?",
        company="Apple",
        year=2024,
        top_k=5
    )

    print(f"Number of results returned: {len(results)}\n")


    for index, document in enumerate(results, start=1):

        print("-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print("Content:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    vector_store.close()


if __name__ == "__main__":
    main()