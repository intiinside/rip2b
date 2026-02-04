"""
Preparación y carga del corpus Amazon Consumer Reviews
Limita el dataset a 1000-10000 productos
"""

import pandas as pd
import os
from PIL import Image
import requests
from io import BytesIO


def cargar_dataset(ruta_csv, limite=15000):
    """
    Carga el dataset de Amazon y limita el número de productos
    
    Args:
        ruta_csv: ruta al archivo CSV del dataset
        limite: número máximo de productos a cargar (entre 1000-10000)
    
    Returns:
        DataFrame con los productos
    """
    print(f"Cargando dataset desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv)
    
    # Limitar dataset
    df = df.head(limite)
    
    # Eliminar duplicados por nombre de producto
    df = df.drop_duplicates(subset=['name'], keep='first')
    
    print(f"Dataset cargado: {len(df)} productos")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    return df
'''
def cargar_dataset(ruta_csv, limite=15000, max_reseñas=20):
    print(f"Cargando dataset desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv)
    
    # Limitar dataset
    df = df.head(limite)
    
    # Mantener máximo 'max_reseñas' reseñas por producto
    df = df.groupby('name').head(max_reseñas)
    
    print(f"Dataset cargado: {len(df)} reseñas distribuidas en {df['name'].nunique()} productos")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    return df'''


def limpiar_datos(df):
    """
    Limpia y prepara los datos del dataset
    
    Args:
        df: DataFrame con productos
    
    Returns:
        DataFrame limpio
    """
    # Eliminar filas con valores nulos en campos importantes
    df = df.dropna(subset=['name'])
    
    # Crear descripción completa combinando campos disponibles
    df['descripcion_completa'] = df.apply(lambda row: crear_descripcion(row), axis=1)
    
    # Limpiar URLs de imágenes
    if 'imageURLs' in df.columns:
        df['imagen_url'] = df['imageURLs'].apply(extraer_primera_url)
    
    return df


def crear_descripcion(row):
    """
    Crea una descripción completa del producto
    
    Args:
        row: fila del DataFrame
    
    Returns:
        string con descripción completa
    """
    partes = []
    
    if pd.notna(row.get('name')):
        partes.append(f"Producto: {row['name']}")
    
    if pd.notna(row.get('brand')):
        partes.append(f"Marca: {row['brand']}")
    
    if pd.notna(row.get('categories')):
        partes.append(f"Categoría: {row['categories']}")
    
    if pd.notna(row.get('reviews.text')):
        partes.append(f"Reseña: {row['reviews.text'][:200]}")
    
    return " | ".join(partes)


def extraer_primera_url(url_string):
    """
    Extrae la primera URL de una cadena que puede contener múltiples URLs
    
    Args:
        url_string: string con URLs
    
    Returns:
        primera URL válida o None
    """
    if pd.isna(url_string):
        return None
    
    # Si es una lista en formato string
    if url_string.startswith('['):
        urls = url_string.strip('[]').replace('"', '').replace("'", '').split(',')
        return urls[0].strip() if urls else None
    
    return url_string


def descargar_imagen(url, timeout=5):
    """
    Descarga una imagen desde URL
    
    Args:
        url: URL de la imagen
        timeout: tiempo máximo de espera
    
    Returns:
        objeto PIL Image o None si falla
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error descargando imagen: {e}")
    
    return None


def guardar_dataset_procesado(df, ruta_salida):
    """
    Guarda el dataset procesado
    
    Args:
        df: DataFrame procesado
        ruta_salida: ruta donde guardar el CSV
    """
    df.to_csv(ruta_salida, index=False)
    print(f"Dataset procesado guardado en: {ruta_salida}")


if __name__ == "__main__":
    # Ejemplo de uso
    ruta_dataset = "data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
    
    # Cargar y limpiar datos
    df = cargar_dataset(ruta_dataset, limite=5000)
    df = limpiar_datos(df)
    
    # Guardar dataset procesado
    guardar_dataset_procesado(df, "dataset_procesado.csv")
    
    print("\nPrimeros 3 productos:")
    print(df[['name', 'descripcion_completa']].head(3))