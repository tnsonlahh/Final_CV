from qdrant_client import QdrantClient
from helpers import get_text_embedding
import matplotlib.pyplot as plt
from PIL import Image

# ================= CONFIG =================
COLLECTION_NAME = "fashion_clip"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 20
ALPHA = 0.6  # weight cho image, (1 - alpha) cho title

# ================= CONNECT QDRANT =================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
def search_text_to_image_hybrid(query_text: str, top_k: int = TOP_K):
    query_vector = get_text_embedding(query_text).tolist()

    # 1️⃣ Search theo IMAGE embedding
    image_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="image",
        limit=top_k,
    ).points

    # 2️⃣ Search theo TITLE embedding
    title_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="text",
        limit=top_k,
    ).points

    # 3️⃣ Merge scores
    score_dict = {}

    for hit in image_results:
        score_dict[hit.id] = {
            "point": hit,
            "image_score": hit.score,
            "title_score": 0.0
        }

    for hit in title_results:
        if hit.id not in score_dict:
            score_dict[hit.id] = {
                "point": hit,
                "image_score": 0.0,
                "title_score": hit.score
            }
        else:
            score_dict[hit.id]["title_score"] = hit.score

    # 4️⃣ Weighted sum
    for item in score_dict.values():
        item["final_score"] = (
            ALPHA * item["image_score"]
            + (1 - ALPHA) * item["title_score"]
        )

    # 5️⃣ Sort
    ranked = sorted(
        score_dict.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked[:top_k]
def visualize_results(results, n_rows=4, n_cols=5):
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i, item in enumerate(results):
        hit = item["point"]
        image_path = hit.payload["image_path"]
        title = hit.payload.get("title", "No title")

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Cannot load image {image_path}: {e}")
            continue

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    query = "ugly shoes"
    results = search_text_to_image_hybrid(query)
    visualize_results(results)
