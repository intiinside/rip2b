"""
rag_gemini.py - Generación con Gemini (RAG)
"""

import google.generativeai as genai
import pandas as pd


def generar_recomendacion(query, productos_df, api_key):
    """
    Genera recomendación usando RAG con Gemini
    
    Args:
        query: consulta del usuario
        productos_df: DataFrame con productos después de re-ranking
        api_key: API key de Gemini
    
    Returns:
        string con respuesta generada
    """
    genai.configure(api_key=api_key)
    
    # Construir contexto
    contexto = "PRODUCTOS DISPONIBLES:\n\n"
    
    for idx, row in productos_df.head(5).iterrows():
        contexto += f"**Producto {row.get('ranking_reranking', idx+1)}:**\n"
        contexto += f"- Nombre: {row['name']}\n"
        
        if pd.notna(row.get('brand')):
            contexto += f"- Marca: {row['brand']}\n"
        
        if pd.notna(row.get('categories')):
            contexto += f"- Categoría: {row['categories']}\n"
        
        if 'score_reranking' in row:
            contexto += f"- Relevancia: {row['score_reranking']:.3f}\n"
        
   # Prompt ajustado para recomendación con justificación
    prompt = f"""
Eres un asistente de compras que recomienda productos basándote en información recuperada.

CONSULTA DEL USUARIO: {query}

{contexto}

INSTRUCCIONES:
1. Recomienda uno o más productos de la lista.
2. Justifica la recomendación usando atributos del contexto (nombre, marca, categoría, reseñas).
3. No inventes información que no esté en el contexto.
4. Escribe en español, de forma clara y natural.
5. Sé conciso pero convincente, como si aconsejaras a un cliente.

FORMATO:
Recomendación:
- Producto recomendado: [Nombre]
- Justificación: [Explicación basada en atributos/ reseñas/ categoría]
"""


    
    # Generar
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)
    
    return response.text