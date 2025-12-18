import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ================= CONFIG =================
CSV_PATH = "metadata.csv"
COLLECTION_NAME = "fashion_clip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# ================= LOAD CSV =================
df = pd.read_csv(CSV_PATH)

# ================= LOAD CLIP =================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(DEVICE)
model.eval()

# ================= CONNECT QDRANT =================
client = QdrantClient(host="localhost", port=6333)

# ================= CREATE COLLECTION =================
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )

# ================= UPLOAD =================
points = []
point_id = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["image_path"]

    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    vector = emb.cpu().numpy()[0].tolist()

    payload = {
        "product_id": int(row["product_id"]),
        "gender": row["gender"],
        "category": row["category"],
        "sub_category": row["sub_category"],
        "product_type": row["product_type"],
        "colour": row["colour"],
        "usage": row["usage"],
        "title": row["title"],
        "image_path": row["image_path"],
        "image_url": row["image_url"],
    }

    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
    )

    point_id += 1

    if len(points) >= BATCH_SIZE:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        points = []

# Insert remaining
if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("✅ Upload metadata + embeddings to Qdrant DONE")
print("Total points:", client.count(collection_name=COLLECTION_NAME))
