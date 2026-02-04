import torch
import clip
import numpy as np
import pandas as pd

def buscar_por_texto(query, text_model, text_index, df, top_k=20):
    query_emb = text_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = text_index.search(query_emb, top_k)

    resultados = df.iloc[indices[0]].copy()
    resultados["similitud"] = scores[0]
    resultados["ranking_inicial"] = range(1, len(resultados) + 1)
    return resultados

def buscar_por_imagen(imagen, clip_model, preprocess, device, image_index, df, top_k=20):
    img_tensor = preprocess(imagen).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    query_emb = emb.cpu().numpy().astype("float32")
    scores, indices = image_index.search(query_emb, top_k)

    resultados = df.iloc[indices[0]].copy()
    resultados["similitud"] = scores[0]
    resultados["ranking_inicial"] = range(1, len(resultados) + 1)
    return resultados

# 🚀 NUEVO: función para mostrar resultados completos
def mostrar_resultados(resultados, max_resultados=5):
    """
    Muestra los resultados con todos los campos relevantes
    
    Args:
        resultados: DataFrame con los resultados
        max_resultados: número máximo de filas a mostrar
    """
    if resultados is None or len(resultados) == 0:
        print("No se encontraron resultados.")
        return
    
    for i, (_, row) in enumerate(resultados.head(max_resultados).iterrows(), start=1):
        print(f"\nResultado {i}")
        print("ID:", row.get('id'))
        print("Producto:", row.get('name'))
        print("Marca:", row.get('brand'))
        print("Categoría:", row.get('categories'))
        print("Fecha reseña:", row.get('reviews.date'))
        print("Rating:", row.get('reviews.rating'))
        print("Título reseña:", row.get('reviews.title'))
        print("Texto reseña:", row.get('reviews.text'))
        print("Usuario:", row.get('reviews.username'))
        print("Similitud:", row.get('similitud'))
        print("Ranking inicial:", row.get('ranking_inicial'))
        print("-" * 50)
