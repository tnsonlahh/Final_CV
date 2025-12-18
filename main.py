from qdrant_client import QdrantClient
from helpers import get_text_embedding
import matplotlib.pyplot as plt
from PIL import Image
# ================= CONFIG =================
COLLECTION_NAME = "fashion_clip"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 15
# ================= CONNECT QDRANT =================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
def search_text_to_image(query_text: str, top_k: int = TOP_K):
    query_vector = get_text_embedding(query_text)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(), 
        using="image",              
        limit=top_k,
    )

    return response.points

def visualize_results(results, n_rows=3, n_cols=5):
    assert n_rows * n_cols >= len(results), "Grid too small for number of results"

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i, hit in enumerate(results):
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

        # wrap title cho khỏi dài
        plt.title(title, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.show()
# ================= RUN DEMO =================
if __name__ == "__main__":
    query = "men shoes match with jeans"
    results = search_text_to_image(query)
    visualize_results(results)
