"""
indexar_corpus.py - Genera embeddings y crea índices FAISS separados
Texto → SentenceTransformer
Imagen → CLIP
"""

import pandas as pd
import numpy as np
import faiss
import torch
import clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm


# ---------- Config ----------
DATASET_PATH = "dataset_procesado.csv"
TEXT_INDEX_PATH = "faiss_text.index"
IMAGE_INDEX_PATH = "faiss_image.index"
TEXT_EMB_PATH = "text_embeddings.npy"
IMAGE_EMB_PATH = "image_embeddings.npy"


# ---------- Utils ----------
def descargar_imagen(url):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return None


# ---------- Main ----------
def main():
    df = pd.read_csv(DATASET_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== TEXT EMBEDDINGS =====
    print("🔤 Generando embeddings de texto...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    textos = df["descripcion_completa"].fillna("").tolist()
    text_embeddings = text_model.encode(textos, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(TEXT_EMB_PATH, text_embeddings)

    text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
    text_index.add(text_embeddings.astype("float32"))
    faiss.write_index(text_index, TEXT_INDEX_PATH)

    print(f"✅ Índice de texto guardado en {TEXT_INDEX_PATH}")

    # ===== IMAGE EMBEDDINGS =====
    print("🖼️ Generando embeddings de imagen...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    image_embeddings = []
    for url in tqdm(df["imagen_url"].fillna("").tolist()):
        img = descargar_imagen(url)
        if img is None:
            image_embeddings.append(np.zeros(512))
            continue

        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        image_embeddings.append(emb.cpu().numpy()[0])

    image_embeddings = np.array(image_embeddings).astype("float32")
    np.save(IMAGE_EMB_PATH, image_embeddings)

    image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
    image_index.add(image_embeddings)
    faiss.write_index(image_index, IMAGE_INDEX_PATH)

    print(f"✅ Índice de imagen guardado en {IMAGE_INDEX_PATH}")
    print("\n🎉 Indexación completada!")


if __name__ == "__main__":
    main()
