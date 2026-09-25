import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings 
from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from vectorstore.weaviate_vectorstore import WeaviateVectorStore, Retriever
from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics


load_dotenv()


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from aPDF filename.
    
    Supports names like `2024_Apple.pdf` and `2024_AnnualReport_Apple.pdf`.
    """

    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        year = parts[0]
        company = parts[-1]
    elif len(parts) >= 2:
        year = parts[1]
        company = parts[0]
    else:
        year = ""
        company = stem

    return company, year


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store:WeaviateVectorStore
) -> None:
    """
    Ingest a single PDF document.
    Pipeline:
        PDF
        ↓
        Markdown
        ↓
        Semantic chunks
        ↓
        HuggingFace embeddings
        ↓
        Weaviate Cloud
        ↓
        Retriever
        ↓
        KPI extraction
        ↓
        PostgreSQL
    """

    pdf_file = Path(pdf_path)

    company, year = parse_company_year(pdf_file)
    print(f"INgesting {pdf_file.name} as company={company!r}, year={year!r}")

    # Convert PDF to Markdown
    converter = PDFToMarkdownConverter()


    markdown_file = converter.convert_pdf(
        pdf_path= pdf_path,
        output_dir="data/markdown"
    )
    print(f"Markdown created: {markdown_file}")

    # Semantic Chunking
    chunks = chunk_markdown(
        markdown_file = markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks for {pdf_file.name}")

    if not chunks:
        print("No chunks generated. Skipping document.")
        return

    # 4. Store Chunks + vectors in Weaviate
    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name
    )
    print(
        f"Successfully stored {len(chunks)} chunks "
        f"in Weaviate."
    )

    # 5. Create Weaviate retriever 
    retriever = Retriever(vector_store)

    # 6. Extract financial metrics using RAG
    metrics = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=int(year) if year.isdigit() else None
    )

    # 7. Save Extracted metrics to PostgreSQL
    if metrics:
        save_metrics(
            company=company, 
            year=int(year) if str(year).isdigit() else None, 
            metrics=metrics
        )
        print(
            f"Financial metrics saved for "
            f"{company} ({year})."
        )
    else:
        print(
            f"No financial metrics extracted for "
            f"{company} ({year})."
        )

def ingest_directory(input_dir: str) -> None:
    """
    Ingest all PDFs from a directory.
    """

    # 1. Load Local Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name=("sentence-transformers/all-MiniLM-L6-v2")
    )
    print("Embedding model loaded successfully.")

    # 2. Connect to Weaviate cloud
    vector_store = WeaviateVectorStore(
        collection_name = "FinancialDocuments",
        embedding_function=embeddings
    )
    print("Weaviate vector store initialized successfully.")


    # 3.Find PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")

    if not pdf_files:
        print("No PDF files found.")
        vector_store.close()
        return


    # 4. Ingest PDFs
    try:
        for pdf_file in pdf_files:
            print("\n" + "=" * 70)
            print(f"PROCESSING: {pdf_file.name}")
            print("=" * 70)
            ingest_document(
                pdf_path=str(pdf_file),
                embeddings=embeddings,
                vector_store=vector_store
        )

    finally:
        vector_store.close()
        print(f"\nWeaviate connection closed")


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")