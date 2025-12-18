import os
import torch
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from helpers import get_image_embedding, get_text_embedding

# ================= CONFIG =================
CSV_PATH = "metadata_preprocessed.csv"
COLLECTION_NAME = "fashion_clip"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 32

df = pd.read_csv(CSV_PATH)

# ================= CONNECT QDRANT =================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ================= CREATE COLLECTION =================
existing_collections = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": VectorParams(
                size=512,
                distance=Distance.COSINE
            ),
            "text": VectorParams(
                size=512,
                distance=Distance.COSINE
            ),
        }
    )
    print(f"Created collection `{COLLECTION_NAME}`")

# ================= UPLOAD DATA =================
points = []
point_id = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["image_path"]
    title = str(row["title"])

    if not os.path.exists(image_path):
        continue

    # ---- Get embeddings ----
    image_vector = get_image_embedding(image_path)
    text_vector = get_text_embedding(title)

    # ---- Payload (metadata) ----
    payload = {
        "product_id": int(row["product_id"]),
        "gender": row["gender"],
        "category": row["category"],
        "sub_category": row["sub_category"],
        "product_type": row["product_type"],
        "colour": row["colour"],
        "usage": row["usage"],
        "title": title,
        "image_path": image_path,
        "image_url": row["image_url"],
    }

    points.append(
        PointStruct(
            id=point_id,
            vector={
                "image": image_vector.tolist(),
                "text": text_vector.tolist(),
            },
            payload=payload
        )
    )

    point_id += 1

    # ---- Batch upsert ----
    if len(points) >= BATCH_SIZE:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        points = []

# ---- Insert remaining ----
if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("Upload completed")
print("Total points:", client.count(collection_name=COLLECTION_NAME).count)
