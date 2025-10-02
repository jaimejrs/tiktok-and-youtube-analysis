# app.py

# --- 1. Importação das Bibliotecas Principais e dos Módulos ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from collections import Counter
import re
import numpy as np
from scipy.stats import ttest_ind
import nltk
from nltk.corpus import stopwords

# Importando nossas funções dos módulos em src/
from src.data_loader import carregar_dados
from src.analysis import analisar_sentimento, carregar_modelo_sentimento, treinar_modelo_features
from src.plotting import plotar_grafico_linha, plotar_grafico_barra


# --- 2. Configurações Iniciais ---
# (O código de download do NLTK e a configuração da página permanecem aqui)
try:
    stopwords.words('portuguese')
except LookupError:
    st.info("Baixando recursos de linguagem (stopwords)...")
    nltk.download('stopwords')

st.set_page_config(layout="wide", page_title="Análise de Vídeos Virais")

# --- 3. Carregamento Inicial e Filtros ---

# AJUSTE IMPORTANTE: O caminho para o arquivo de dados mudou!
df_original = carregar_dados('data/youtube_shorts_tiktok_trends_2025.csv')

st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("Selecione os Países:", options=sorted(df_original['country'].unique()), default=df_original['country'].unique())
plataformas = st.sidebar.multiselect("Selecione as Plataformas:", options=sorted(df_original['platform'].unique()), default=df_original['platform'].unique())
tipos_dispositivo = st.sidebar.multiselect("Selecione o Device:", options=sorted(df_original['device_type'].unique()), default=df_original['device_type'].unique())

df_filtrado = df_original.query(
    "country == @paises and platform == @plataformas and device_type == @tipos_dispositivo"
)

# --- 4. Construção do Dashboard (UI) ---
# O restante do código, que constrói a interface do Streamlit, permanece aqui sem alterações.
# (Cole o resto do seu código, do st.title() em diante, aqui)

st.title("📊🎦 Análise de Performance de Vídeos Virais")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste sua seleção.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise dos Fatores", "Análise do Conteúdo", "Análise Geográfica"])

    # --- ABA 1: VISÃO GERAL ---
    with tab1:
        st.header("Visão Geral dos Dados")
        # ... (todo o resto do seu código da UI) ...
