from pathlib import Path
from langchain_chroma import Chroma

from types import SimpleNamespace


class ChromaVectorStore:
    """ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "financial_documents",
        embedding_function=None,
        persist_directory: str = "./chroma_db"
    ) -> None:
        """
        Initialize a persistent ChromaDB vector store.

        Args:
            collection_name: Name of the Chroma collection.
            embedding_function: Embedding model used for documents and queries.
            persist_directory: Local directory where ChromaDB is stored.
        """

        if embedding_function is None:
            raise ValueError(
                "embedding_function must be provided."
            )

        self.client = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to ChromaDB.
        """
        documents = []

        for chunk in chunks:
            chunk.metadata.update({
                "company": company,
                "year": year,
                "source_file": source_file
            })

            documents.append(chunk)

        if documents:
            self.client.add_documents(documents)

        print(
            f"Uploaded {len(documents)} chunks to ChromaDB."
        )
        

class Retriever:
    """
    Wrapper around the LangChain Chroma retriever.

    Keeps a similar interface to the old Azure Retriever.
    """
    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Retrieve relevant documents from ChromaDB

        Optional company/year filters are applied through
        Chroma metadata filtering.
        """

        search_kwargs = {
            "k": top_k
        }

        filters = {}

        if company:
            filters["company"] = company

        if year is not None:
            filters["year"] = str(year)

        if filters:
            if len(filters) == 1:
                search_kwargs["filter"] = filters
            else:
                search_kwargs["filter"] = {
                    "$and": [
                        {"company": company},
                        {"year": str(year)}
                    ]
                }

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

        results = retriever.invoke(query)

        return results