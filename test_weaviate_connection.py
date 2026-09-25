import os

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth


load_dotenv()

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

if not weaviate_url:
    raise ValueError("WEAVIATE_URL is not set in .env")

if not weaviate_api_key:
    raise ValueError("WEAVIATE_API_KEY is not set in .env")


print("Connecting to Weaviate Cloud...")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

try:
    if client.is_ready():
        print("SUCCESS: Weaviate Cloud is ready.")
    else:
        print("ERROR: Weaviate Cloud is not ready.")

finally:
    client.close()