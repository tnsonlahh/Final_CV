from qdrant_client import QdrantClient
from helpers import get_text_embedding, visualize_results
# ================= CONFIG =================
COLLECTION_NAME = "fashion_clip"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 20
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

# ================= RUN DEMO =================
if __name__ == "__main__":
    query = "ugly shoes"
    results = search_text_to_image(query)
    visualize_results(results)
