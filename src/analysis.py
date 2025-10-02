# src/analysis.py

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline

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

    for col in df_model.select_dtypes(include=['object']).columns:
        df_model[col] = df_model[col].astype('category').cat.codes

    X = df_model[features_model]
    y = df_model[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
