import weaviate
from typing import List, Dict

WEAVIATE_URL = "http://localhost:8080"

def get_client():
    return weaviate.Client(WEAVIATE_URL)


def create_schema():
    """
    Create the schema for DocumentChunk class.
    Run once before upserting anything.
    """
    client = get_client()

    schema = {
        "classes": [
            {
                "class": "DocumentChunk",
                "vectorIndexType": "hnsw",
                "vectorizer": "none",
                "properties": [
                    { "name": "text", "dataType": ["text"] },
                    { "name": "doc_id", "dataType": ["string"] },
                    { "name": "page", "dataType": ["int"] }
                ]
            }
        ]
    }

    client.schema.delete_all()     # optional — clears DB for fresh start
    client.schema.create(schema)


def vector_upsert(objects: List[Dict]):
    """
    objects = [
        {
            "id": "doc123_p1_c0",
            "vector": [0.12, 0.8, ...],
            "properties": {
                "text": "...",
                "doc_id": "doc123",
                "page": 1
            }
        },
        ...
    ]
    """
    client = get_client()

    with client.batch(batch_size=20) as batch:
        for obj in objects:
            batch.add_data_object(
                data_object=obj["properties"],
                class_name="DocumentChunk",
                uuid=obj["id"],
                vector=obj["vector"]
            )
