"""
app.py - Interfaz Streamlit con diseño moderno inspirado en el HTML
Sistema de Recuperación Multimodal E-Commerce
"""

import streamlit as st
import pandas as pd
import torch
import clip
import faiss
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer
import google.generativeai as genai
from io import BytesIO
import requests
import os

from retrieval import buscar_por_texto, buscar_por_imagen
from reranking import aplicar_reranking
from rag_gemini import generar_recomendacion
from busqueda_conversacional import (
    inicializar_estado,
    aplicar_filtros,
    actualizar_historial,
    resetear_busqueda
)
# Detectar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Añadir aquí BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Cargar BLIP directamente en memoria real (CPU/GPU)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False   # evita meta tensors
)

# Si quieres GPU, usa accelerate con device_map="auto"
# pero si no lo tienes instalado, déjalo en CPU
if device == "cuda":
    try:
        import accelerate
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32,
            device_map="auto"
        )
    except ImportError:
        print("⚠️ 'accelerate' no está instalado, el modelo se queda en CPU.")

def generar_descripcion(imagen):
    inputs = processor(images=imagen, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)





# Configuración de página
st.set_page_config(
    page_title="Sistema RAG Multimodal",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================= CSS MEJORADO (Basado en el HTML) =========================
st.markdown("""
<style>
    /* No external font import, to use the monospace font from the theme */

    /* Variables de color para el tema Hacker */
    :root {
        --primary-color: #0D0D2B;
        --background-color: #0A0A1F;
        --secondary-background-color: #1A1A3D;
        --text-color: #FFFFFF;
        --accent-color: #00FFFF; /* Un cian brillante para acentos */
        --hacker-green: #39FF14;
    }
    
    /* Fuente global */
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'monospace', sans-serif !important;
    }
    
    /* Fondo principal */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Contenedor principal */
    .main .block-container {
        background-color: var(--primary-color);
        border: 1px solid var(--accent-color);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        padding: 40px;
        margin: 40px auto;
        max-width: 1200px;
    }
    
    /* Título principal */
    .main-title {
        font-size: 2.8em;
        font-weight: 700;
        text-shadow: 0 0 10px var(--accent-color), 0 0 20px var(--accent-color);
        color: var(--accent-color);
        text-align: center;
        margin-bottom: 10px;
        padding: 20px 0;
    }
    
    /* Subtítulo */
    .subtitle {
        text-align: center;
        color: #CCCCCC;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    /* Badge de estado */
    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: var(--secondary-background-color);
        color: var(--hacker-green);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 10px auto;
        border: 1px solid var(--hacker-green);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--hacker-green);
        border-radius: 50%;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 5px var(--hacker-green); opacity: 1; }
        50% { box-shadow: 0 0 15px var(--hacker-green); opacity: 0.7; }
    }
    
    /* Tabs estilizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        color: var(--text-color);
        padding: 0 25px;
        font-size: 1.1em;
        border: 1px solid var(--accent-color);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--primary-color);
        border-color: var(--hacker-green);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-color) !important;
        color: var(--primary-color) !important;
        border: none;
        box-shadow: 0 0 15px var(--accent-color);
    }
    
    /* Botones estilizados */
    .stButton > button {
        background: transparent !important;
        color: var(--accent-color) !important;
        border: 2px solid var(--accent-color) !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--accent-color) !important;
        color: var(--primary-color) !important;
        box-shadow: 0 0 20px var(--accent-color) !important;
    }
    
    /* Input de texto */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid var(--accent-color) !important;
        padding: 12px 20px !important;
        font-size: 1.1em !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--hacker-green) !important;
        box-shadow: 0 0 15px var(--hacker-green) !important;
    }
    
    .stTextInput > label {
        font-weight: 600;
        color: var(--accent-color);
        font-size: 1.1em;
    }
    
    /* File uploader mejorado */
    .stFileUploader {
        border: 2px dashed var(--accent-color) !important;
        border-radius: 10px !important;
        padding: 40px 20px !important;
        text-align: center;
        background-color: var(--secondary-background-color);
    }
    
    .stFileUploader:hover {
        border-color: var(--hacker-green) !important;
    }
    
    /* Cards de productos */
    .product-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid #2A2A4D;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        border-color: var(--accent-color);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--secondary-background-color);
        border-radius: 10px;
        font-weight: 600;
        color: var(--text-color);
        border: 1px solid var(--accent-color);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.8em;
        font-weight: 700;
        color: var(--hacker-green);
        text-shadow: 0 0 5px var(--hacker-green);
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #CCCCCC;
    }
    
    /* Divisor horizontal */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent 0%, var(--accent-color) 50%, transparent 100%);
        opacity: 0.5;
        margin: 40px 0;
    }
    
    /* Alertas */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid var(--accent-color);
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--primary-color);
        border-right: 1px solid var(--accent-color);
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--hacker-green);
    }
    
</style>
""", unsafe_allow_html=True)


# ========================= CARGA DEL SISTEMA =========================
@st.cache_resource
def cargar_sistema():
    """Carga todos los recursos del sistema una sola vez"""
    try:
        df = pd.read_csv("dataset_procesado.csv")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Modelos ---
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        text_model = SentenceTransformer("all-MiniLM-L6-v2")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # --- Índices FAISS ---
        text_index = faiss.read_index("faiss_text.index")
        image_index = faiss.read_index("faiss_image.index")

        return {
            "df": df,
            "clip_model": clip_model,
            "preprocess": preprocess,
            "device": device,
            "text_model": text_model,
            "text_index": text_index,
            "image_index": image_index,
            "cross_encoder": cross_encoder,
            "estado": "ok"
        }
    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}


# ========================= UI COMPONENTS =========================
def mostrar_producto_card(producto, posicion):
    """Muestra un producto en formato card con diseño mejorado"""
    st.markdown('<div class="product-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    
   # Imagen del producto
    with col1:
        image_urls = producto.get("imageURLs")

        if pd.notna(image_urls) and str(image_urls).strip() != "":
            try:
                # Tomar SOLO la primera imagen
                url = str(image_urls).split(",")[0].strip()
                st.image(url, use_container_width=True)
            except:
                st.markdown("<p style='text-align:center;'>📦 Sin imagen</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center;'>📦 Sin imagen</p>", unsafe_allow_html=True)


    # Información del producto
    with col2:
        st.markdown(f"### {posicion}. {producto.get('name','')[:80]}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if pd.notna(producto.get("brand")):
                st.markdown(f"**🏷️ Marca:** {producto['brand']}")
        with col_b:
            if "score_reranking" in producto:
                st.metric("Score", f"{producto['score_reranking']:.3f}")
        with col_c:
            if pd.notna(producto.get("reviews.rating")):
                st.markdown(f"**⭐ Rating:** {producto['reviews.rating']}")

        if pd.notna(producto.get("categories")):
            st.caption(f"📂 {producto['categories'][:100]}")

        # 🔥 Campos adicionales de reseña
        if pd.notna(producto.get("reviews.title")):
            st.markdown(f"**📝 Título reseña:** {producto['reviews.title']}")
        if pd.notna(producto.get("reviews.text")):
            st.markdown(f"**💬 Reseña:** {producto['reviews.text'][:300]}...")
        if pd.notna(producto.get("reviews.username")):
            st.caption(f"👤 Usuario: {producto['reviews.username']}")
        if pd.notna(producto.get("reviews.date")):
            st.caption(f"📅 Fecha: {producto['reviews.date']}")

    st.markdown("</div>", unsafe_allow_html=True)



# ========================= MAIN =========================
def main():
    # Header principal con diseño del HTML
    st.markdown("""
        <div class="content-header">
            <h1 class="main-title">✨PROMPT MARKET </h1>
            <p class="subtitle">Encuentre el mejor producto del mercado usando texto o imagen con IA</p>
        </div>
    """, unsafe_allow_html=True)

    sistema = cargar_sistema()

    if sistema["estado"] == "error":
        st.error(f"❌ Error al cargar el sistema: {sistema['mensaje']}")
        st.info("💡 Asegúrate de haber ejecutado los scripts de indexación")
        return

    # Badge de estado activo
    st.markdown("""
        <div style="text-align: center;">
            <div class="status-badge">
                <div class="status-dot"></div>
                Sistema Activo
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if "state" not in st.session_state:
        st.session_state.state = inicializar_estado()
   



    top_k = 20
    top_rerank = 5
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

    # ===================== TABS CON ICONOS =====================
    tab_texto, tab_imagen = st.tabs(["🔍 Buscar por Texto", "🖼️ Buscar por Imagen"])

    # ---------- TAB: BÚSQUEDA POR TEXTO ----------
    with tab_texto:
       


        st.markdown("""
            <div class="content-header">
                <h3>🔍 Encuentra productos escribiendo tu consulta</h3>
                <p>Describe el producto que buscas y obtén recomendaciones personalizadas</p>
            </div>
        """, unsafe_allow_html=True)

        query = st.text_input(
            "Tu búsqueda",
            placeholder="Pregunta lo que necesitas",
            label_visibility="visible",
            key="query_text"
        )

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            buscar_texto = st.button("🔍 Buscar", type="primary", use_container_width=True, key="btn_text")

        if buscar_texto:
            if not query:
                st.warning("⚠️ Por favor, escribe una consulta de búsqueda")
            else:
                with st.spinner("🔎 Buscando productos relevantes..."):
                    resultados = buscar_por_texto(
                        query,
                        sistema["text_model"],
                        sistema["text_index"],
                        sistema["df"],
                        top_k
                    )

                    resultados_final = aplicar_reranking(
                        query, resultados, sistema["cross_encoder"], top_rerank
                    )

                    st.session_state.state["resultados"] = resultados_final
                    st.session_state.state["query"] = query

            '''    # --- Generar recomendación IA (solo si hay api_key) ---
            if api_key:
                    respuesta = generar_recomendacion(query, resultados_final, api_key)
                    st.session_state.state["respuesta_ia"] = respuesta

                    # Actualizar historial
                    actualizar_historial(st.session_state.state, query)


        # --- Mostrar recomendación IA arriba de los resultados ---
        if st.session_state.state.get("respuesta_ia") and st.session_state.state.get("resultados") is not None:
            st.markdown("### 🧠 Recomendación IA")
            st.info(st.session_state.state["respuesta_ia"])'''


 
    # ---------- TAB: BÚSQUEDA POR IMAGEN ----------
    with tab_imagen:
       

        st.markdown("""
            <div class="content-header">
                <h3>🖼️ Encuentra productos similares desde una imagen</h3>
                <p>Sube una foto del producto que te interesa y encuentra opciones similares</p>
            </div>
        """, unsafe_allow_html=True)

        col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
        with col_upload2:
            uploaded = st.file_uploader(
                "📤 Arrastra o selecciona una imagen",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="visible",
                key="image_upload"
            )

        if uploaded:
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                imagen = Image.open(uploaded).convert("RGB")
                st.image(imagen, caption="📸 Tu imagen", width=300)

            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
            with col_btn2:
                buscar_imagen = st.button("🔍 Buscar Similares", type="primary", use_container_width=True, key="btn_image")

            if buscar_imagen:
                with st.spinner("🔎 Analizando imagen y buscando productos similares..."):
                    resultados = buscar_por_imagen(
                        imagen,
                        sistema["clip_model"],
                        sistema["preprocess"],
                        sistema["device"],
                        sistema["image_index"],
                        sistema["df"],
                        top_k
                    )

                    # ✅ Generar descripción automática de la imagen 
                    descripcion_imagen = generar_descripcion(imagen)
                    query_rerank = descripcion_imagen
                    
                    resultados_final = aplicar_reranking(
                        query_rerank, resultados, sistema["cross_encoder"], top_rerank
                    )

                    st.session_state.state["resultados"] = resultados_final
                    st.session_state.state["query"] = "🖼️ Búsqueda por imagen"
                    
                '''     # --- Generar recomendación IA (solo si hay api_key) ---
                if api_key:
                    respuesta = generar_recomendacion(query, resultados_final, api_key)
                    st.session_state.state["respuesta_ia"] = respuesta

                    # Actualizar historial
                    actualizar_historial(st.session_state.state, query)


        # --- Mostrar recomendación IA arriba de los resultados ---
        if st.session_state.state.get("respuesta_ia") and st.session_state.state.get("resultados") is not None:
            st.markdown("### 🧠 Recomendación IA")
            st.info(st.session_state.state["respuesta_ia"])'''


        # ===================== MOSTRAR RESULTADOS =====================
    if st.session_state.state.get("resultados") is not None:
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style='text-align: center; margin: 30px 0;'>
                <h3>🔍 Resultados para: <em>{st.session_state.state['query']}</em></h3>
            </div>
        """, unsafe_allow_html=True)

        # SOLO PRODUCTOS (sin tab de Recomendación IA)
        st.markdown("#### 🏆 Productos Más Relevantes")
        resultados = st.session_state.state["resultados"]

        for idx, row in resultados.iterrows():
            mostrar_producto_card(row, row.get("ranking_reranking", idx + 1))

        # Métricas finales
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Productos mostrados", len(resultados))
        with col2:
            st.metric("🔍 Recuperados inicialmente", top_k)
        with col3:
            if len(resultados) > 0 and "score_reranking" in resultados.iloc[0]:
                avg_score = resultados["score_reranking"].mean()
                st.metric("⭐ Score promedio", f"{avg_score:.3f}")



if __name__ == "__main__":
    main()
    