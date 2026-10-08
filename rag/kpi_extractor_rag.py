import os

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, Field

from llm.groq_client import get_structured_completion
from vectorstore.weaviate_vectorstore import WeaviateVectorStore, Retriever
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


def retrieve_context(retriever: Retriever, company: str, year: int) -> str:
    queries = [
    "consolidated statements of income net income revenue operating income",
    "consolidated statements of cash flows operating activities",
    "consolidated balance sheets total assets total liabilities",
    "net income",
    "total assets total liabilities",
    "cash flows from operating activities",
    "risk factors",
    "growth drivers"
]

    all_documents = []

    for query in queries:
        documents = retriever.invoke(
            query=query,
            company=company,
            year=year,
            top_k=3
        )
        all_documents.extend(documents)

    # Remove duplicate chunks
    unique_documents = []
    seen = set()

    for document in all_documents:
        content = document.page_content.strip()

        if content and content not in seen:
            seen.add(content)
            unique_documents.append(document)

    MAX_TOTAL_CHARS = 24000
    MAX_CHARS_PER_DOCUMENT = 3000

    context_parts = []
    total_chars = 0

    for document in unique_documents:
        content = document.page_content.strip()[:MAX_CHARS_PER_DOCUMENT]

        remaining = MAX_TOTAL_CHARS - total_chars

        if remaining <= 0:
            break

        content = content[:remaining]
        context_parts.append(content)
        total_chars += len(content)

    print(
        f"Retrieved {len(unique_documents)} unique documents; "
        f"using {len(context_parts)} chunks and {total_chars} characters of context."
    )

    print("\n===== RETRIEVED CONTEXT PREVIEW =====")
    for i, part in enumerate(context_parts, 1):
        print(f"\n--- CHUNK {i} ---")
        print(part[:1000])
    print("===== END RETRIEVED CONTEXT =====\n")

    return "\n\n---\n\n".join(context_parts)


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    return f"""
Extract financial information from the provided {company} {year} annual-report context.

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null if unavailable.
- Financial values must match the report exactly.
- Risk factors should be concise.
- Growth drivers should be concise.
- Return valid JSON only.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract financial KPIs using RAG and Groq.
    """
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )
    if not context:
        return {}

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )

    return metrics.model_dump(
        by_alias=True
    )


def main() -> None:
    """
    Test KPI extraction directly
    """
    company = "Apple"
    year = 2024

    print("=" * 70)
    print("Loading embedding model")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded successfully.\n")

    print("=" * 70)
    print("Connecting to Weaviate")
    print("=" * 70)

    vector_store = WeaviateVectorStore(
        collection_name="FinancialDocuments",
        embedding_function=embeddings
    ) 
    print("Weaviate vector store initialized successfully.\n")

    try:
        # IMPORTANT:
        # Retriever expects the WeaviateVectorStore,
        # not vector_store.client.
        retriever = Retriever(vector_store)

        print("=" * 70)
        print(f"Extracting KPIs for {company} {year}")
        print("=" * 70)

        results = extract_financial_metrics(
            retriever=retriever,
            company=company,
            year=year
        )
        print(f"\nExtracted KPIs for {company} {year}\n")

        if not results:
            print("No KPIs were extracted.")
            return

        for key, value in results.items():
            print(f"{key}:")
            print(value)
            print("-" * 80)

        # -----------------------------------------------------
        # Save extracted metrics to PostgreSQL
        # -----------------------------------------------------
        from database.save_metrics import save_metrics

        save_metrics(
            company=company,
            year=year,
            metrics=results
        )
        print(f"\nFinancial metrics saved to PostgreSQL.")

    finally:
        vector_store.close()
        print("\nWeaviate connection closed.")

if __name__ == "__main__":
    main()