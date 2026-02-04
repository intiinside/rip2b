"""
reranking.py - Re-ranking con cross-encoder
"""

from sentence_transformers import CrossEncoder
import pandas as pd


def aplicar_reranking(query, resultados, cross_encoder, top_k=10):
    """
    Aplica re-ranking a los resultados usando cross-encoder
    
    Args:
        query: consulta del usuario
        resultados: DataFrame con resultados del retrieval
        cross_encoder: modelo cross-encoder cargado
        top_k: número de resultados finales
    
    Returns:
        DataFrame reordenado con columna score_reranking y ranking_reranking
    """
    # Crear pares query-documento
    pares = []
    for _, row in resultados.iterrows():
        texto = row.get('descripcion_completa', row.get('name', ''))
        pares.append([query, texto])
    
    # Calcular scores
    scores = cross_encoder.predict(pares)
    
    # Reordenar
    resultados_reranked = resultados.copy()
    resultados_reranked['score_reranking'] = scores
    resultados_reranked = resultados_reranked.sort_values('score_reranking', ascending=False)
    resultados_reranked['ranking_reranking'] = range(1, len(resultados_reranked) + 1)
    
    return resultados_reranked.head(top_k)