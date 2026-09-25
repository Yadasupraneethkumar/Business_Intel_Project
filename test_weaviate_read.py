import weaviate
import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 70)
    print("Connecting to Weaviate")
    print("=" * 70)

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.Auth.api_key(
            os.getenv("WEAVIATE_API_KEY")
        )
    )

    try:
        print("Connected:", client.is_ready())

        collection = client.collections.use("FinancialDocuments")

        print("\n" + "=" * 70)
        print("Reading objects from FinancialDocuments")
        print("=" * 70)

        response = collection.query.fetch_objects(
            limit=10
        )

        print(f"Number of objects found: {len(response.objects)}")

        for i, obj in enumerate(response.objects, start=1):
            print("\n" + "-" * 70)
            print(f"OBJECT {i}")
            print("-" * 70)
            print("UUID:", obj.uuid)
            print("Properties:", obj.properties)

    finally:
        client.close()
        print("\nWeaviate connection closed.")


if __name__ == "__main__":
    main()