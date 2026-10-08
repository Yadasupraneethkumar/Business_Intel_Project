from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from vectorstore.weaviate_vectorstore import (
    WeaviateVectorStore,
    Retriever
)


PDF_PATH = "data/raw_pdfs/2024_Apple.pdf"
MARKDOWN_DIR = "data/markdown"


def main():

    print("=" * 70)
    print("STEP 1: Checking PDF")
    print("=" * 70)

    pdf_file = Path(PDF_PATH)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print(f"PDF found: {pdf_file}")
    print()

    # ---------------------------------------------------------
    # STEP 2: Load embedding model
    # ---------------------------------------------------------

    print("=" * 70)
    print("STEP 2: Loading embedding model")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded successfully.")
    print()

    # ---------------------------------------------------------
    # STEP 3: Convert PDF to Markdown
    # ---------------------------------------------------------

    print("=" * 70)
    print("STEP 3: Converting PDF to Markdown")
    print("=" * 70)

    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        pdf_path=PDF_PATH,
        output_dir=MARKDOWN_DIR
    )

    print(f"Markdown file created: {markdown_file}")
    print()

    # ---------------------------------------------------------
    # STEP 4: Semantic chunking
    # ---------------------------------------------------------

    print("=" * 70)
    print("STEP 4: Creating semantic chunks")
    print("=" * 70)

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Number of chunks generated: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No chunks were generated from the PDF."
        )

    print()

    # ---------------------------------------------------------
    # STEP 5: Connect to Weaviate
    # ---------------------------------------------------------

    print("=" * 70)
    print("STEP 5: Connecting to Weaviate")
    print("=" * 70)

    vector_store = WeaviateVectorStore(
        collection_name="FinancialDocuments",
        embedding_function=embeddings
    )

    print("Connected to Weaviate successfully.")
    print()

    try:

        # -----------------------------------------------------
        # STEP 6: Upload chunks
        # -----------------------------------------------------

        print("=" * 70)
        print("STEP 6: Uploading chunks to Weaviate")
        print("=" * 70)

        vector_store.upload_chunks(
            chunks=chunks,
            embeddings=embeddings,
            company="Apple",
            year="2024",
            source_file=pdf_file.name
        )

        print(
            f"Uploaded {len(chunks)} chunks successfully."
        )
        print()

        # -----------------------------------------------------
        # STEP 7: Test retrieval
        # -----------------------------------------------------

        print("=" * 70)
        print("STEP 7: Testing retrieval")
        print("=" * 70)

        retriever = Retriever(vector_store)

        results = retriever.invoke(
            query="What was Apple's revenue?",
            company="Apple",
            year=2024,
            top_k=5
        )

        print(
            f"Number of retrieved documents: {len(results)}"
        )

        for index, document in enumerate(
            results,
            start=1
        ):
            print()
            print("-" * 70)
            print(f"RESULT {index}")
            print("-" * 70)

            print(
                document.page_content[:1000]
            )

            print()
            print("Metadata:")
            print(document.metadata)

        if not results:
            print()
            print(
                "WARNING: No documents were retrieved."
            )

    finally:

        vector_store.close()

        print()
        print("Weaviate connection closed.")

    print()
    print("=" * 70)
    print("REAL PDF WEAVIATE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()





    