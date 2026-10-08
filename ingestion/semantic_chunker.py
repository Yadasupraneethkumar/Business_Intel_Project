from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content from a file.

    Args:
        markdown_file: Path to the markdown file.

    Returns:
        Markdown content as a string.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown content.

    Args:
        markdown_file: Path to the markdown file.
        embeddings: Embedding model used for semantic chunking.

    Returns:
        List of LangChain Document objects containing
        semantically split chunks.
    """

    markdown_content = read_markdown(markdown_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
        ],
    )

    return splitter.create_documents([markdown_content])


if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    # Use the same embedding model used by ChromaDB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    markdown_file = "../data/markdown/2024_Apple.md"

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()