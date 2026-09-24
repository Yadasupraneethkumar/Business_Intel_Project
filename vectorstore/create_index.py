from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_collection(
    collection_name: str = "financial_documents",
    persist_directory: str = "./chroma_db"
) -> None:
    """
    Create or open a persistent ChromaDB collection.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    print(
        f"ChromaDB collection '{collection_name}' "
        f"is ready."
    )


if __name__ == "__main__":
    create_collection()