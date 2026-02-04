# Sistema de Recuperación Multimodal de Información para E-commerce

Este proyecto implementa un sistema de **Recuperación de Información Multimodal** aplicado a un escenario de comercio electrónico. El sistema integra búsqueda semántica (texto e imagen), un mecanismo de re-ranking para refinar la relevancia y Generación Aumentada por Recuperación (RAG) para ofrecer respuestas explicativas y justificadas al usuario.

El proyecto fue desarrollado como parte de **Recuperación de Información** (Prof. Iván Carrera).

## 📋 Descripción del Proyecto

El objetivo principal es diseñar un motor de búsqueda que permita a los usuarios encontrar productos utilizando consultas de texto o imágenes. El sistema no solo recupera productos, sino que "entiende" la intención del usuario manteniendo el contexto de la conversación y generando respuestas en lenguaje natural que justifican por qué se recomiendan dichos productos.

### Funcionalidades Principales

* **Indexación Multimodal:** Codificación de imágenes y descripciones de productos en un espacio vectorial común utilizando modelos como CLIP.
* **Búsqueda Híbrida:** Soporte para consultas *Text-to-Product* e *Image-to-Product*.
* **Re-ranking:** Refinamiento de los resultados iniciales (Top-k) mediante modelos Cross-Encoder o similitud refinada para mejorar la precisión.
* **RAG (Retrieval-Augmented Generation):** Generación de respuestas contextualizadas utilizando LLMs (ej. GPT-4 o Gemini), fundamentadas en la evidencia recuperada (título, atributos, reseñas) .
* **Búsqueda Conversacional:** Memoria de sesión que permite refinar búsquedas (ej. "muéstrame estos pero en color rojo") sin perder el contexto del producto anterior .
* **Interfaz Gráfica:** UI interactiva desarrollada en **Streamlit/Gradio** para visualizar resultados y chat.

## 🛠️ Arquitectura del Pipeline

El flujo de información sigue las etapas de un sistema de RI moderno:

1. **Retrieval Inicial:** Búsqueda vectorial aproximada (ANN) sobre un índice (FAISS/ChromaDB) para obtener candidatos.
2. **Re-ranking:** Reordenamiento de los candidatos basado en una evaluación más costosa pero precisa de la relevancia.
3. **Generación (RAG):** Construcción de un prompt con el contexto de los productos top-rankeados y generación de la respuesta final.

## 🚀 Instalación y Configuración

### Prerrequisitos

* Python 3.8+ 


* Cuenta de OpenAI o Google (para la API de generación).
* Dataset de Amazon Products (Kaggle).

### 1. Clonar el repositorio

```bash
git clone https://github.com/intiinside/rip2b.git
cd rip2b

```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt

```

*(Asegúrate de incluir en requirements.txt librerías como: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `streamlit`, `openai`, `pandas`, etc.)*

### 4. Configuración de Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto para tus claves de API:

```env
OPENAI_API_KEY="tu_clave_aqui"
# O si usas Gemini
GOOGLE_API_KEY="tu_clave_aqui"

```

## 📂 Preparación del Dataset

El proyecto utiliza el corpus **Consumer Reviews of Amazon Products**.

1. Descarga el dataset desde Kaggle.
2. Coloca las imágenes y los archivos de metadatos en la carpeta `data/raw/`.
3. 
**Nota:** Para facilitar la ejecución local, se utiliza un subconjunto de productos (1,000 - 10,000 items).



Para procesar e indexar los datos, ejecuta:

```bash
python scripts/index_data.py

```

*Este script genera los embeddings y guarda el índice vectorial en `data/indices/`.*

## 💻 Ejecución del Sistema

Para iniciar la interfaz de usuario conversacional:

```bash
streamlit run app.py

```

O si utilizaste Gradio:

```bash
python app.py

```

El sistema estará disponible en `http://localhost:8501` (o el puerto indicado en la consola).

🗂️ Estructura del Proyecto 

```text
├── data/
│   ├── raw/                # Dataset original (imágenes y csv)
│   └── indices/            # Índices vectoriales (FAISS/Chroma) guardados
├── src/
│   ├── retrieval.py        # Lógica de búsqueda vectorial (Embeddings CLIP)
│   ├── rerank.py           # Lógica de re-ranking (Cross-Encoders)
│   ├── generation.py       # Módulo RAG (Conexión con LLM)
│   └── utils.py            # Funciones auxiliares de carga de datos
├── app.py                  # Interfaz de usuario (Streamlit/Gradio)
├── scripts/
│   └── index_data.py       # Script para crear los índices
├── notebooks/              # Análisis exploratorio y pruebas
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
└── .env                    # Variables de entorno (no subir al repo)

```

## 📊 Ejemplos de Uso

### Ejemplo 1: Búsqueda Texto-Imagen

* **Usuario:** "Zapatos deportivos para correr en montaña"
* **Sistema:** Recupera imágenes de zapatillas de trail running, reordena las mejores opciones y explica: *"Te recomiendo el modelo X por su suela antideslizante mencionada en las reseñas..."*

Ejemplo 2: Refinamiento Conversacional 

* **Usuario:** (Sube una foto de una chaqueta azul) "Busco algo así".
* **Sistema:** Muestra chaquetas similares.
* **Usuario:** "Mejor en color negro y más barata".
* **Sistema:** Filtra los resultados anteriores buscando variantes negras y de menor precio, manteniendo el estilo original.

## ✒️ Autores

* Inti Poaquiza Azogue 
* Angel Falcon

---

**Nota sobre la Evaluación:** Este proyecto incluye un informe técnico (`docs/informe.pdf`) con el análisis cualitativo del impacto del re-ranking y métricas de recuperación, según lo solicitado en la rúbrica .