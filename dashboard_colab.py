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

st.set_page_config(layout="wide", page_title="Análise de Vídeos Virais")

@st.cache_data
def carregar_dados():
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
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data
def analisar_sentimento(_df, _pipeline):
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

df_original = carregar_dados()

st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("Selecione os Países:", options=sorted(df_original['country'].unique()), default=df_original['country'].unique())
plataformas = st.sidebar.multiselect("Selecione as Plataformas:", options=sorted(df_original['platform'].unique()), default=df_original['platform'].unique())
tipos_dispositivo = st.sidebar.multiselect("Selecione o Device:", options=sorted(df_original['device_type'].unique()), default=df_original['device_type'].unique())

df_filtrado = df_original.query("country == @paises and platform == @plataformas and device_type == @tipos_dispositivo")

st.title("📊🎦 Análise de Performance de Vídeos Virais")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste sua seleção.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise dos Fatores", "Análise do Conteúdo", "Análise Geográfica"])

    with tab1:
        st.header("Visão Geral dos Dados")
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
                fig = px.bar(importancias_filtradas, y=importancias_filtradas.index, x=importancias_filtradas.values, orientation='h', color=importancias_filtradas.index)
                fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'}, xaxis_title='Importância Relativa', yaxis_title='Fator')
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Taxa de Engajamento por Tipo de Dispositivo")
        engagement_by_device = df_filtrado.groupby('device_type')['engagement_rate'].mean().sort_values(ascending=False).reset_index()
        fig_device = px.bar(engagement_by_device, x='device_type', y='engagement_rate', color='device_type', labels={'device_type': 'Tipo de Dispositivo', 'engagement_rate': 'Taxa de Engajamento Média'}, log_y=True)
        fig_device.update_layout(showlegend=False)
        st.plotly_chart(fig_device, use_container_width=True)

    with tab2:
        st.header("Análise de Fatores de Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Engajamento por Hora de Upload")
            engagement_by_hour = df_filtrado.groupby('upload_hour')['engagement_rate'].mean().reset_index()
            fig = px.line(engagement_by_hour, x='upload_hour', y='engagement_rate', markers=True, labels={'upload_hour': 'Hora do Dia (24h)', 'engagement_rate': 'Taxa de Engajamento Média'})
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Engajamento Total Mediano por Categoria")
            engagement_by_category = df_filtrado.groupby('category')['engagement_total'].median().sort_values(ascending=False)
            fig = px.bar(engagement_by_category, x=engagement_by_category.index, y=engagement_by_category.values, color=engagement_by_category.index, labels={'x': 'Categoria', 'y': 'Engajamento Mediano'})
            fig.update_layout(yaxis_type="log", showlegend=False, bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Engajamento por Duração do Vídeo")
            bins = [0, 15, 30, 60, 120, np.inf]
            labels = ['0-15s', '16-30s', '31-60s', '61-120s', '120s+']
            df_filtrado_copy = df_filtrado.copy()
            df_filtrado_copy['duration_bin'] = pd.cut(df_filtrado_copy['duration_sec'], bins=bins, labels=labels, right=False)
            engagement_by_duration = df_filtrado_copy.groupby('duration_bin')['engagement_rate'].mean().reset_index()
            fig = px.bar(engagement_by_duration, x='duration_bin', y='engagement_rate', color='duration_bin', labels={'duration_bin': 'Faixa de Duração', 'engagement_rate': 'Taxa de Engajamento Média'})
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Engajamento por Dia da Semana")
            engagement_by_weekday = df_filtrado.groupby('publish_dayofweek')['engagement_rate'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig = px.bar(engagement_by_weekday, x='publish_dayofweek', y='engagement_rate', color='publish_dayofweek', labels={'publish_dayofweek': 'Dia da Semana', 'engagement_rate': 'Taxa de Engajamento Média'}, log_y=True)
            fig.update_layout(showlegend=False, bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Análise do Conteúdo dos Vídeos")
        st.subheader("Top 10 Palavras-chave (Alto Engajamento)")
        high_engagement_threshold = df_filtrado['engagement_rate'].quantile(0.75)
        df_high_engagement = df_filtrado[df_filtrado['engagement_rate'] >= high_engagement_threshold]
        all_keywords = " ".join(df_high_engagement['title_keywords'].dropna())
        words = re.findall(r'\b\w+\b', all_keywords.lower())
        stopwords = ['a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'os', 'as','â','for','in','on','is','i','to']
        filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
        word_counts = Counter(filtered_words)
        top_words = pd.DataFrame(word_counts.most_common(10), columns=['Palavra', 'Frequência'])
        fig = px.bar(top_words, x='Frequência', y='Palavra', orientation='h', color='Palavra')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Análise de Sentimento dos Comentários")
        sentiment_pipeline = carregar_modelo_sentimento()
        with st.spinner("Analisando sentimentos (amostra de 500)... Isso pode levar um minuto."):
            df_sentimento = analisar_sentimento(df_filtrado, sentiment_pipeline)
        if not df_sentimento.empty:
            sentiment_distribution = df_sentimento.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            fig = go.Figure()
            colors = {'Positivo': 'seagreen', 'Neutro': 'gold', 'Negativo': 'tomato'}
            for sentiment in sentiment_distribution.columns:
                fig.add_trace(go.Bar(name=sentiment, x=sentiment_distribution.index, y=sentiment_distribution[sentiment], marker_color=colors.get(sentiment)))
            fig.update_layout(barmode='stack', title='Distribuição de Sentimento por Categoria', xaxis_title='Categoria', yaxis_title='Número de Comentários')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há comentários para analisar com os filtros atuais.")
        st.subheader("Teste A/B: Engajamento com vs. Sem Emoji no Título")
        com_emoji = df_filtrado[df_filtrado['has_emoji'] == 1]['engagement_rate'].dropna()
        sem_emoji = df_filtrado[df_filtrado['has_emoji'] == 0]['engagement_rate'].dropna()
        if not com_emoji.empty and not sem_emoji.empty:
            hist_data = [com_emoji, sem_emoji]
            group_labels = ['Com Emoji', 'Sem Emoji']
            colors = ['#EF553B', '#636EFA']
            fig = ff.create_distplot(hist_data, group_labels, bin_size=.005, colors=colors, show_rug=False)
            fig.update_layout(title_text='Distribuição da Taxa de Engajamento', xaxis_title='Taxa de Engajamento', yaxis_title='Densidade')
            st.plotly_chart(fig, use_container_width=True)
            stat, p_value = ttest_ind(com_emoji, sem_emoji, equal_var=False)
            col1_metric, col2_metric = st.columns(2)
            col1_metric.metric(label="Estatística do Teste T", value=f"{stat:.4f}")
            col2_metric.metric(label="P-valor", value=f"{p_value:.4f}")
        else:
            st.info("Não há dados suficientes para realizar o Teste A/B com os filtros atuais.")

    with tab4:
        st.header("Análise Geográfica")
        st.subheader("Performance por País (Visualizações vs. Engajamento)")
        analise_paises = df_filtrado.groupby('country').agg(avg_views=('views', 'mean'), avg_engagement_rate=('engagement_rate', 'mean'), video_count=('row_id', 'count')).reset_index()
        fig = px.scatter(
            analise_paises, x='avg_views', y='avg_engagement_rate',
            size='video_count', color='country', hover_name='country',
            log_x=True, size_max=60, text='country',
            labels={"avg_views": "Média de Visualizações (Log)", "avg_engagement_rate": "Taxa de Engajamento Média"})
        fig.update_traces(textposition='middle center', textfont=dict(color='white'))
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
            st.info("Não há dados suficientes para criar o heatmap de Categoria e Região com os filtros atuais.")


!git config --global user.name "JaimeJrs"
!git config --global user.email "jaimetjribeiro@gmail.com"
