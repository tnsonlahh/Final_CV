import os
import gradio as gr
from PIL import Image
from qdrant_client import QdrantClient
from helpers import get_text_embedding

# ================= CONFIG =================
COLLECTION_NAME = "fashion_clip"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 6

# ================= CONNECT QDRANT =================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ================= SEARCH FUNCTION =================
def search_fashion_images(text_query, top_k=TOP_K):
    if not text_query.strip():
        return []

    # 1. Encode text
    query_vector = get_text_embedding(text_query)

    # 2. Search Qdrant (TEXT VECTOR SPACE)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        using="image",
        limit=top_k
    ).points

    image_paths = []
    images = []

    for hit in hits:
        payload = hit.payload
        img_path = payload.get("image_path")

        if img_path and os.path.exists(img_path):
            image_paths.append(img_path)
            images.append(Image.open(img_path).convert("RGB"))

    return images

# ================= GRADIO UI =================
with gr.Blocks(title="Fashion Recommendation System") as demo:
    gr.Markdown("## Fashion Image Recommendation (CLIP + Qdrant)")
    gr.Markdown("Nhập mô tả sản phẩm, hệ thống trả về ảnh phù hợp")

    with gr.Row():
        text_input = gr.Textbox(
            label="Text query",
            placeholder="Ví dụ: white sneakers for men"
        )

    with gr.Row():
        search_btn = gr.Button("Search")

    with gr.Row():
        gallery_output = gr.Gallery(
            label="Recommended images",
            columns=3,
            height=320
        )

    search_btn.click(
        fn=search_fashion_images,
        inputs=text_input,
        outputs= gallery_output
    )

demo.launch(share=True)
