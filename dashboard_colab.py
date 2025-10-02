# dashboard_colab_refatorado.py

# --- 1. Importação das Bibliotecas ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from collections import Counter
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
from scipy.stats import ttest_ind
import nltk
from nltk.corpus import stopwords

# --- 2. Configurações Iniciais e Funções de Cache ---

# Download de recursos necessários do NLTK (executado apenas uma vez)
try:
    stopwords.words('portuguese')
except LookupError:
    st.info("Baixando recursos de linguagem (stopwords)...")
    nltk.download('stopwords')

st.set_page_config(layout="wide", page_title="Análise de Vídeos Virais")

@st.cache_data
def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV, tratando datas e criando colunas temporais."""
    df = pd.read_csv(
        caminho_arquivo,
        encoding='utf-8',  # Padronizado para UTF-8 para maior compatibilidade.
        delimiter=',',
        parse_dates=['publish_date_approx']
    )
    df['year_month'] = df['publish_date_approx'].dt.to_period('M').astype(str)
    df['publish_dayofweek'] = df['publish_date_approx'].dt.day_name()
    return df

@st.cache_resource
def carregar_modelo_sentimento():
    """Carrega o modelo de análise de sentimento da Hugging Face."""
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data
def analisar_sentimento(_df: pd.DataFrame, _pipeline) -> pd.DataFrame:
    """Aplica o modelo de análise de sentimento em uma amostra de comentários."""
    df_comentarios = _df.dropna(subset=['sample_comments'])
    if df_comentarios.empty:
        return pd.DataFrame()

    sample_size = min(len(df_comentarios), 500)
    df_sample = df_comentarios.sample(sample_size, random_state=42)

    def get_sentiment(comment: str) -> str:
        try:
            result = _pipeline(comment[:512])[0]
            score = int(result['label'].split(' ')[0])
            if score > 3: return 'Positivo'
            if score < 3: return 'Negativo'
            return 'Neutro'
        # Captura exceções específicas para melhor depuração.
        except (IndexError, KeyError, RuntimeError):
            return 'N/A'

    df_sample['sentiment'] = df_sample['sample_comments'].apply(get_sentiment)
    return df_sample

@st.cache_data
def treinar_modelo_features(_df: pd.DataFrame) -> pd.Series:
    """Treina um RandomForest para extrair a importância das features."""
    if _df.shape[0] < 10:
        return pd.Series(dtype='float64')

    features_model = ['duration_sec', 'upload_hour', 'is_weekend', 'category', 'creator_tier']
    target = 'engagement_rate'
    df_model = _df[features_model + [target]].copy().dropna()

    if df_model.empty:
        return pd.Series(dtype='float64')

    # Codificação de categóricos de forma idiomática com pandas
    for col in df_model.select_dtypes(include=['object']).columns:
        df_model[col] = df_model[col].astype('category').cat.codes

    X = df_model[features_model]
    y = df_model[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# --- 3. Funções Auxiliares de Plotagem (Princípio DRY) ---

def plotar_grafico_linha(df, x_col, y_col, agg_func, titulo, **kwargs):
    """Função reutilizável para criar gráficos de linha agregados."""
    df_agg = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
    fig = px.line(df_agg, x=x_col, y=y_col, markers=True, title=titulo, **kwargs)
    st.plotly_chart(fig, use_container_width=True)

def plotar_grafico_barra(df, x_col, y_col, titulo, **kwargs):
    """Função reutilizável para criar gráficos de barra."""
    fig = px.bar(df, x=x_col, y=y_col, title=titulo, **kwargs)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. Carregamento Inicial e Filtros ---

df_original = carregar_dados('youtube_shorts_tiktok_trends_2025.csv')

st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("Selecione os Países:", options=sorted(df_original['country'].unique()), default=df_original['country'].unique())
plataformas = st.sidebar.multiselect("Selecione as Plataformas:", options=sorted(df_original['platform'].unique()), default=df_original['platform'].unique())
tipos_dispositivo = st.sidebar.multiselect("Selecione o Device:", options=sorted(df_original['device_type'].unique()), default=df_original['device_type'].unique())

# Utiliza o método .query() para um código mais legível.
df_filtrado = df_original.query(
    "country == @paises and platform == @plataformas and device_type == @tipos_dispositivo"
)

# --- 5. Construção do Dashboard ---

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
            # Evita cópia desnecessária do DataFrame.
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

        # Utiliza NLTK para uma lista de stopwords mais robusta.
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
            # Normaliza para exibir em percentual (stack de 100%)
            sentiment_distribution_norm = sentiment_distribution.div(sentiment_distribution.sum(axis=1), axis=0)
            
            # Gráfico de barras empilhadas com Plotly Express (mais conciso).
            fig = px.bar(sentiment_distribution_norm, 
                         barmode='stack', 
                         color_discrete_map={'Positivo': 'seagreen', 'Neutro': 'gold', 'Negativo': 'tomato'},
                         title='Distribuição Percentual de Sentimento por Categoria',
                         labels={'value': 'Proporção', 'category': 'Categoria'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há comentários para analisar com os filtros atuais.")

        st.subheader("Teste A/B: Engajamento com vs. Sem Emoji no Título")
        # Usando .query() para clareza
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
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Taxa de Engajamento por Categoria e Região")
        pivot_engagement = df_filtrado.pivot_table(values='engagement_rate', index='region', columns='category', aggfunc='mean')
        if not pivot_engagement.empty:
            fig = px.imshow(pivot_engagement, text_auto=".3f", aspect="auto",
                            labels=dict(x="Categoria", y="Região", color="Engajamento Médio"),
                            color_continuous_scale='YlGnBu')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há dados suficientes para criar o heatmap com os filtros atuais.")
