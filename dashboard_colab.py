# dashboard_colab.py

# --- 1. Importação das Bibliotecas ---
# Aqui, eu importo todas as ferramentas que vou precisar para o meu projeto.
# O Streamlit é a base do meu dashboard. O Pandas é essencial para manipular os dados.
# Plotly, a minha escolha para criar os gráficos interativos.
# Scikit-learn e Transformers são para as análises mais avançadas de Machine Learning.
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from collections import Counter
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from scipy.stats import ttest_ind

# --- 2. Configuração da Página ---
# Defino que meu dashboard usará o layout "wide" (tela cheia) para melhor aproveitamento do espaço.
# Também dou um título que aparecerá na aba do navegador.
st.set_page_config(layout="wide", page_title="Análise de Vídeos Virais")

# --- 3. Funções de Processamento de Dados com Cache ---
# Para otimizar a performance, eu crio funções específicas para tarefas pesadas
# e uso os "decoradores" de cache do Streamlit (@st.cache_data e @st.cache_resource).
# Isso significa que essas operações só serão executadas uma vez, tornando o dashboard muito mais rápido.

@st.cache_data
def carregar_dados():
    """
    Minha função principal para carregar e preparar os dados.
    Ela lê o arquivo CSV e cria colunas de data que serão úteis para as análises temporais.
    O @st.cache_data garante que o CSV só seja lido uma vez.
    """
    df = pd.read_csv(
        'youtube_shorts_tiktok_trends_2025.csv',
        encoding='ISO-8859-1',
        delimiter=',',
        parse_dates=['publish_date_approx']
    )
    df['year_month'] = df['publish_date_approx'].dt.to_period('M').astype(str)
    df['publish_dayofweek'] = df['publish_date_approx'].dt.day_name()
    return df

@st.cache_resource
def carregar_modelo_sentimento():
    """
    Esta função carrega o modelo de análise de sentimento da Hugging Face.
    É uma operação muito pesada, então o @st.cache_resource é perfeito para garantir
    que o modelo seja carregado na memória apenas na primeira vez que o app inicia.
    """
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data
def analisar_sentimento(_df, _pipeline):
    """
    Aqui, eu aplico o modelo de sentimento nos comentários.
    Para não sobrecarregar o app, pego uma amostra de 500 comentários.
    A função interna 'get_sentiment' classifica cada comentário como Positivo, Negativo ou Neutro.
    """
    if _df.empty: return pd.DataFrame()
    sample_size = min(len(_df[_df['sample_comments'].notna()]), 500)
    if sample_size == 0: return pd.DataFrame()
    df_sample = _df[_df['sample_comments'].notna()].sample(sample_size, random_state=42)
    def get_sentiment(comment):
        try:
            result = _pipeline(comment[:512])[0]
            score = int(result['label'].split(' ')[0])
            if score > 3: return 'Positivo'
            elif score < 3: return 'Negativo'
            else: return 'Neutro'
        except Exception: return 'N/A'
    df_sample['sentiment'] = df_sample['sample_comments'].apply(get_sentiment)
    return df_sample

@st.cache_resource
def treinar_modelo_features(_df):
    """
    Nesta função, eu treino um modelo de Machine Learning (Random Forest) não para prever,
    mas para descobrir quais fatores (features) são mais importantes para explicar
    a taxa de engajamento. O @st.cache_resource armazena o modelo treinado.
    """
    if _df.empty or _df.shape[0] < 10: return pd.Series()
    features_model = ['duration_sec', 'upload_hour', 'is_weekend', 'category', 'creator_tier']
    target = 'engagement_rate'
    df_model = _df[features_model + [target]].copy().dropna()
    if df_model.empty: return pd.Series()
    for col in ['category', 'creator_tier']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
    X = df_model[features_model]
    y = df_model[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances

# --- 4. Carregamento Inicial e Filtros ---
# Executo a função para carregar os dados. O resultado fica guardado na variável 'df_original'.
df_original = carregar_dados()

# Crio a barra lateral do meu dashboard, onde coloco todos os filtros interativos.
st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("Selecione os Países:", options=sorted(df_original['country'].unique()), default=df_original['country'].unique())
plataformas = st.sidebar.multiselect("Selecione as Plataformas:", options=sorted(df_original['platform'].unique()), default=df_original['platform'].unique())
tipos_dispositivo = st.sidebar.multiselect("Selecione o Device:", options=sorted(df_original['device_type'].unique()), default=df_original['device_type'].unique())

# A mágica acontece aqui: eu filtro o DataFrame original com base nas seleções do usuário.
# Todos os gráficos do painel usarão este 'df_filtrado'.
df_filtrado = df_original.query("country == @paises and platform == @plataformas and device_type == @tipos_dispositivo")

# --- 5. Construção do Dashboard ---
# Adiciono o título principal do meu projeto.
st.title("📊🎦 Análise de Performance de Vídeos Virais")

# Crio uma verificação de segurança: se os filtros resultarem em nenhum dado,
# eu exibo um aviso em vez de tentar desenhar os gráficos e gerar um erro.
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste sua seleção.")
else:
    # Se houver dados, eu crio as abas para organizar meu storytelling.
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise dos Fatores", "Análise do Conteúdo", "Análise Geográfica"])

    # --- ABA 1: VISÃO GERAL ---
    with tab1:
        st.header("Visão Geral dos Dados")
        # Divido a aba em duas colunas para um layout 2x2.
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tendência Mensal de Visualizações")
            df_tendencia_mensal = df_filtrado.groupby('year_month')['views'].sum().reset_index()
            fig = px.line(df_tendencia_mensal, x='year_month', y='views', markers=True, labels={'year_month': 'Mês', 'views': 'Total de Visualizações'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Distribuição do Engajamento por Plataforma")
            engagement_by_platform = df_filtrado.groupby('platform')['engagement_rate'].mean().reset_index()
            fig = px.pie(engagement_by_platform, values='engagement_rate', names='platform', title='Taxa de Engajamento Média', hole=.3)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Tendência Mensal da Taxa de Engajamento")
            df_tendencia_mensal_eng = df_filtrado.groupby('year_month')['engagement_rate'].mean().reset_index()
            fig = px.line(df_tendencia_mensal_eng, x='year_month', y='engagement_rate', markers=True, labels={'year_month': 'Mês', 'engagement_rate': 'Taxa de Engajamento Média'}, color_discrete_sequence=['green'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Importância dos Fatores para o Engajamento")
            importancias_filtradas = treinar_modelo_features(df_filtrado)
            if not importancias_filtradas.empty:
