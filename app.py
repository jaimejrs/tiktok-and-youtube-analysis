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
try:
    stopwords.words('portuguese')
except LookupError:
    st.info("Baixando recursos de linguagem (stopwords)...")
    nltk.download('stopwords')

st.set_page_config(layout="wide", page_title="Análise de Vídeos Virais")

# --- 3. Carregamento Inicial e Filtros ---

df_original = carregar_dados('data/youtube_shorts_tiktok_trends_2025.csv')

st.sidebar.header("Filtros")

# --- FILTRO DE PAÍSES (multiselect com "Selecionar Todos") ---
todos_paises_options = sorted(df_original['country'].unique())
selecionar_todos_paises = st.sidebar.checkbox("Selecionar Todos os Países", value=True)

if selecionar_todos_paises:
    paises_selecionados = st.sidebar.multiselect(
        "Selecione os Países:",
        options=todos_paises_options,
        default=todos_paises_options
    )
else:
    paises_selecionados = st.sidebar.multiselect(
        "Selecione os Países:",
        options=todos_paises_options
    )

# --- FILTRO DE PLATAFORMAS (multiselect com "Selecionar Todas") ---
todas_plataformas_options = sorted(df_original['platform'].unique())
selecionar_todas_plataformas = st.sidebar.checkbox("Selecionar Todas as Plataformas", value=True)

if selecionar_todas_plataformas:
    plataformas_selecionadas = st.sidebar.multiselect(
        "Selecione as Plataformas:",
        options=todas_plataformas_options,
        default=todas_plataformas_options
    )
else:
    plataformas_selecionadas = st.sidebar.multiselect(
        "Selecione as Plataformas:",
        options=todas_plataformas_options
    )

# --- FILTRO DE DISPOSITIVOS ---
todos_dispositivos_options = sorted(df_original['device_type'].unique())
selecionar_todos_dispositivos = st.sidebar.checkbox("Selecionar Todos os Dispositivos", value=True)

if selecionar_todos_dispositivos:
    dispositivos_selecionados = st.sidebar.multiselect(
        "Selecione o Device:",
        options=todos_dispositivos_options,
        default=todos_dispositivos_options
    )
else:
    dispositivos_selecionados = st.sidebar.multiselect(
        "Selecione o Device:",
        options=todos_dispositivos_options
    )

# --- LÓGICA DE FILTRAGEM ---
# A query funciona perfeitamente com as listas geradas pelos multiselects
df_filtrado = df_original.query(
    "country == @paises_selecionados and platform == @plataformas_selecionadas and device_type == @dispositivos_selecionados"
)
# --- 4. Construção do Dashboard ---
st.title("📊🎦 Análise de Performance de Vídeos Virais")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste sua seleção.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise dos Fatores", "Análise do Conteúdo", "Análise Geográfica"])

    # --- ABA 1: VISÃO GERAL ---
    with tab1:
        st.header("Visão Geral dos Dados")
        col1, col2 = st.columns(2)
        with col1:
            plotar_grafico_linha(df_filtrado, 'year_month', 'views', 'sum', 'Tendência Mensal de Visualizações', labels={'year_month': 'Mês', 'views': 'Total de Visualizações'})
            
            engagement_by_platform = df_filtrado.groupby('platform')['engagement_rate'].mean().reset_index()
            fig = px.pie(engagement_by_platform, values='engagement_rate', names='platform', title='Taxa de Engajamento Média', hole=.3)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            plotar_grafico_linha(df_filtrado, 'year_month', 'engagement_rate', 'mean', 'Tendência Mensal da Taxa de Engajamento', labels={'year_month': 'Mês', 'engagement_rate': 'Taxa de Engajamento Média'}, color_discrete_sequence=['green'])

            importancias_filtradas = treinar_modelo_features(df_filtrado)
            if not importancias_filtradas.empty:
                plotar_grafico_barra(importancias_filtradas, importancias_filtradas.values, importancias_filtradas.index, 'Importância dos Fatores para o Engajamento', orientation='h', color=importancias_filtradas.index, labels={'x': 'Importância Relativa', 'y': 'Fator'})

    # --- ABA 2: ANÁLISE DOS FATORES ---
    with tab2:
        st.header("Análise de Fatores de Performance")
        col1, col2 = st.columns(2)
        with col1:
            plotar_grafico_linha(df_filtrado, 'upload_hour', 'engagement_rate', 'mean', 'Engajamento por Hora de Upload', labels={'upload_hour': 'Hora do Dia (24h)', 'engagement_rate': 'Taxa de Engajamento Média'})
            
            engagement_by_category = df_filtrado.groupby('category')['engagement_total'].median().sort_values(ascending=False)
            plotar_grafico_barra(engagement_by_category, engagement_by_category.index, engagement_by_category.values, 'Engajamento Total Mediano por Categoria', color=engagement_by_category.index, labels={'x': 'Categoria', 'y': 'Engajamento Mediano'}, log_y=True)

        with col2:
            bins = [0, 15, 30, 60, 120, np.inf]
            labels = ['0-15s', '16-30s', '31-60s', '61-120s', '120s+']
            df_filtrado['duration_bin'] = pd.cut(df_filtrado['duration_sec'], bins=bins, labels=labels, right=False)
            engagement_by_duration = df_filtrado.groupby('duration_bin', observed=True)['engagement_rate'].mean().reset_index()
            plotar_grafico_barra(engagement_by_duration, 'duration_bin', 'engagement_rate', 'Engajamento por Duração do Vídeo', color='duration_bin', labels={'duration_bin': 'Faixa de Duração', 'engagement_rate': 'Taxa de Engajamento Média'})

            dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            engagement_by_weekday = df_filtrado.groupby('publish_dayofweek')['engagement_rate'].mean().reindex(dias_ordem).reset_index()
            plotar_grafico_barra(engagement_by_weekday, 'publish_dayofweek', 'engagement_rate', 'Engajamento por Dia da Semana', color='publish_dayofweek', labels={'publish_dayofweek': 'Dia da Semana', 'engagement_rate': 'Taxa de Engajamento Média'}, log_y=True)

    # --- ABA 3: ANÁLISE DO CONTEÚDO ---
    with tab3:
        st.header("Análise do Conteúdo dos Vídeos")
        st.subheader("Top 10 Palavras-chave (Alto Engajamento)")
        
        high_engagement_threshold = df_filtrado['engagement_rate'].quantile(0.75)
        df_high_engagement = df_filtrado.query("engagement_rate >= @high_engagement_threshold")
        
        all_keywords = " ".join(df_high_engagement['title_keywords'].dropna())
        words = re.findall(r'\b\w+\b', all_keywords.lower())

        stop_words_pt = set(stopwords.words('portuguese'))
        stop_words_en = set(stopwords.words('english'))
        all_stopwords = stop_words_pt.union(stop_words_en).union(['â','for','in','on','is','i','to'])

        filtered_words = [word for word in words if word not in all_stopwords and not word.isdigit()]
        word_counts = Counter(filtered_words)
        top_words = pd.DataFrame(word_counts.most_common(10), columns=['Palavra', 'Frequência'])
        
        fig = px.bar(top_words, x='Frequência', y='Palavra', orientation='h', color='Palavra')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Análise de Sentimento dos Comentários")
        sentiment_pipeline = carregar_modelo_sentimento()
        with st.spinner("Analisando sentimentos (amostra)..."):
            df_sentimento = analisar_sentimento(df_filtrado, sentiment_pipeline)

        if not df_sentimento.empty:
            sentiment_distribution = df_sentimento.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            sentiment_distribution_norm = sentiment_distribution.div(sentiment_distribution.sum(axis=1), axis=0)
            
            fig = px.bar(sentiment_distribution_norm, 
                         barmode='stack', 
                         color_discrete_map={'Positivo': 'seagreen', 'Neutro': 'gold', 'Negativo': 'tomato'},
                         title='Distribuição Percentual de Sentimento por Categoria',
                         labels={'value': 'Proporção', 'category': 'Categoria'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há comentários para analisar com os filtros atuais.")

        st.subheader("Teste A/B: Engajamento com vs. Sem Emoji no Título")
        com_emoji = df_filtrado.query("has_emoji == 1")['engagement_rate'].dropna()
        sem_emoji = df_filtrado.query("has_emoji == 0")['engagement_rate'].dropna()

        if not com_emoji.empty and not sem_emoji.empty:
            fig = ff.create_distplot([com_emoji, sem_emoji], ['Com Emoji', 'Sem Emoji'], bin_size=.005, colors=['#EF553B', '#636EFA'], show_rug=False)
            fig.update_layout(title_text='Distribuição da Taxa de Engajamento', xaxis_title='Taxa de Engajamento', yaxis_title='Densidade')
            st.plotly_chart(fig, use_container_width=True)

            stat, p_value = ttest_ind(com_emoji, sem_emoji, equal_var=False)
            col1_metric, col2_metric = st.columns(2)
            col1_metric.metric(label="Estatística do Teste T", value=f"{stat:.4f}")
            col2_metric.metric(label="P-valor", value=f"{p_value:.4f}", help="P-valor < 0.05 geralmente indica uma diferença estatisticamente significativa.")
        else:
            st.info("Não há dados suficientes para realizar o Teste A/B com os filtros atuais.")

    # --- ABA 4: ANÁLISE GEOGRÁFICA ---
    with tab4:
        st.header("Análise Geográfica")
        st.subheader("Performance por País (Visualizações vs. Engajamento)")
        
        analise_paises = df_filtrado.groupby('country').agg(
            avg_views=('views', 'mean'), 
            avg_engagement_rate=('engagement_rate', 'mean'), 
            video_count=('row_id', 'count')
        ).reset_index()

        fig = px.scatter(
            analise_paises, x='avg_views', y='avg_engagement_rate',
            size='video_count', color='country', hover_name='country',
            log_x=True, size_max=60, text='country',
            labels={"avg_views": "Média de Visualizações (Log)", "avg_engagement_rate": "Taxa de Engajamento Média"}
        )
        
        fig.update_traces(
            textposition='middle center', 
            textfont=dict(color='white')
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Taxa de Engajamento por Categoria e Região")
        pivot_engagement = df_filtrado.pivot_table(values='engagement_rate', index='region', columns='category', aggfunc='mean')
        if not pivot_engagement.empty:
            fig_heatmap = px.imshow(pivot_engagement, text_auto=".3f", aspect="auto",
                            labels=dict(x="Categoria", y="Região", color="Engajamento Médio"),
                            color_continuous_scale='YlGnBu')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Não há dados suficientes para criar o heatmap com os filtros atuais.")
