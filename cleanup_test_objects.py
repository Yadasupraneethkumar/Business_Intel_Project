import os

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter


load_dotenv()


def main():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(
            os.getenv("WEAVIATE_API_KEY")
        )
    )

    try:
        collection = client.collections.use(
            "FinancialDocuments"
        )

        print("=" * 70)
        print("Finding synthetic test objects")
        print("=" * 70)

        response = collection.query.fetch_objects(
            filters=Filter.by_property(
                "sourceFile"
            ).equal("test_2024_Apple.pdf"),
            limit=100
        )

        print(
            f"Found {len(response.objects)} "
            "synthetic test object(s)."
        )

        for obj in response.objects:
            print(
                f"Deleting {obj.uuid}..."
            )

            collection.data.delete_by_id(
                obj.uuid
            )

        print("\nCleanup completed.")

    finally:
        client.close()
        print("Weaviate connection closed.")


if __name__ == "__main__":
    main()