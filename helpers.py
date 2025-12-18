import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

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

