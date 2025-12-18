import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MODEL ONCE =================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=True)

model.to(DEVICE)
model.eval()


def get_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy()[0]


def get_text_embedding(text: str):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy()[0]

def visualize_results(results, n_rows=4, n_cols=5):
    """
    Display retrieved images in a grid with product titles
    """
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