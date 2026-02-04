"""
busqueda_conversacional.py - Gestión de estado y contexto
"""

import pandas as pd


def inicializar_estado():
    """Crea un nuevo estado de sesión"""
    return {
        'resultados': None,
        'query': '',
        'respuesta_ia': None,
        'filtros': {},
        'historial': []
    }


def aplicar_filtros(resultados_df, filtros):
    """
    Aplica filtros sobre los resultados
    
    Args:
        resultados_df: DataFrame con productos
        filtros: dict con filtros {'color': 'black', 'marca': 'Sony'}
    
    Returns:
        DataFrame filtrado
    """
    df_filtrado = resultados_df.copy()
    
    for tipo, valor in filtros.items():
        valor_lower = valor.lower()
        
        if tipo == 'color':
            # Buscar en nombre y descripción
            mask = (
                df_filtrado['name'].str.lower().str.contains(valor_lower, na=False) |
                df_filtrado.get('descripcion_completa', pd.Series([''] * len(df_filtrado)))
                .str.lower().str.contains(valor_lower, na=False)
            )
            df_filtrado = df_filtrado[mask]
        
        elif tipo == 'marca':
            if 'brand' in df_filtrado.columns:
                mask = df_filtrado['brand'].str.lower().str.contains(valor_lower, na=False)
                df_filtrado = df_filtrado[mask]
    
    return df_filtrado


def actualizar_historial(estado, query):
    """Agrega consulta al historial"""
    estado['historial'].append(query)
    if len(estado['historial']) > 5:
        estado['historial'] = estado['historial'][-5:]


def resetear_busqueda():
    """Reinicia la búsqueda"""
    return inicializar_estado()