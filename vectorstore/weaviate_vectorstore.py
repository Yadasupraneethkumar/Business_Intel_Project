import os

import weaviate
from dotenv import load_dotenv
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject

load_dotenv()


class WeaviateVectorStore:
    """Weaviate Cloud vector store using externally generated embeddings."""

    def __init__(
        self,
        collection_name: str = "FinancialDocuments",
        embedding_function=None
    ) -> None:
        """
        Initialize a connection to Weaviate Cloud.

        Args:
            collection_name: Name of the Weaviate collection.
            embedding_function: Embedding model used for documents and queries.
        """

        if embedding_function is None:
            raise ValueError(
                "embedding_function must be provided."
            )

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url:
            raise ValueError(
                "WEAVIATE_URL is not set in the .env file."
            )

        if not weaviate_api_key:
            raise ValueError(
                "WEAVIATE_API_KEY is not set in the .env file."
            )

        self.embedding_function = embedding_function

        print("Connecting to Weaviate Cloud...")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key)
        )

        if not self.client.is_ready():
            self.client.close()
            raise RuntimeError(
                "Weaviate Cloud is not ready."
            )

        self.collection_name = collection_name

        self._create_collection_if_needed()

        self.collection = self.client.collections.use(
            self.collection_name
        )

        print(
            f"Connected to Weaviate collection: "
            f"{self.collection_name}"
        )

    def _create_collection_if_needed(self) -> None:
        """Create the collection if it does not already exist."""

        if self.client.collections.exists(self.collection_name):
            return

        print(
            f"Creating Weaviate collection "
            f"'{self.collection_name}'..."
        )

        self.client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="company",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="year",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="sourceFile",
                    data_type=DataType.TEXT
                )
            ]
        )

        print(
            f"Created Weaviate collection "
            f"'{self.collection_name}'."
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
        Upload document chunks and their embeddings to Weaviate.

        Each chunk becomes one Weaviate object.
        """

        if not chunks:
            print("No chunks to upload.")
            return

        objects = []

        for chunk in chunks:

            chunk.metadata.update({
                "company": company,
                "year": str(year),
                "source_file": source_file
            })

            content = chunk.page_content

            vector = embeddings.embed_query(content)

            objects.append(
                DataObject(
                    properties= {
                        "content": content,
                        "company": company,
                        "year": str(year),
                        "sourceFile": source_file
                },
                vector = vector
            )
        )

        response = self.collection.data.insert_many(
            objects
        )

        if response.has_errors:
            print(
                "Some objects failed to upload to Weaviate."
            )

            for error in response.errors:
                print(error)

            raise RuntimeError(
                "Weaviate upload failed for one or more chunks."
            )

        print(
            f"Uploaded {len(objects)} chunks to Weaviate."
        )

    def close(self) -> None:
        """Close the Weaviate connection."""

        if self.client:
            self.client.close()


class Retriever:
    """
    Retriever for Weaviate Cloud.

    Performs vector similarity search using the same
    HuggingFace embedding model used during ingestion.
    """

    def __init__(self, vector_store: WeaviateVectorStore) -> None:
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list[Document]:
        """
        Retrieve documents using vector similarity.

        Optional company/year filters can be applied.
        """

        # ---------------------------------------------------------
        # 1. Convert the query into the same embedding space
        # ---------------------------------------------------------
        query_vector = self.vector_store.embedding_function.embed_query(
            query
        )

        # ---------------------------------------------------------
        # 2. Build optional metadata filters
        # ---------------------------------------------------------
        filters = []

        if company:
            filters.append(
                Filter.by_property("company").equal(company)
            )

        if year is not None:
            filters.append(
                Filter.by_property("year").equal(str(year))
            )

        combined_filter = None

        if len(filters) == 1:
            combined_filter = filters[0]

        elif len(filters) > 1:
            combined_filter = filters[0] & filters[1]

        # ---------------------------------------------------------
        # 3. Perform vector similarity search
        # ---------------------------------------------------------
        if combined_filter is not None:
            response = self.vector_store.collection.query.near_vector(
                near_vector=query_vector,
                filters=combined_filter,
                limit=top_k
            )
        else:
            response = self.vector_store.collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k
            )

        # ---------------------------------------------------------
        # 4. Convert Weaviate objects back into LangChain Documents
        # ---------------------------------------------------------
        results = []

        for obj in response.objects:
            properties = obj.properties

            document = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "company": properties.get("company"),
                    "year": properties.get("year"),
                    "source_file": properties.get("sourceFile")
                }
            )

            results.append(document)

        return results