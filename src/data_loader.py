# src/data_loader.py

import pandas as pd
import streamlit as st

@st.cache_data
def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV, tratando datas e criando colunas temporais."""
    df = pd.read_csv(
        caminho_arquivo,
        encoding='utf-8',
        delimiter=',',
        parse_dates=['publish_date_approx']
    )
    df['year_month'] = df['publish_date_approx'].dt.to_period('M').astype(str)
    df['publish_dayofweek'] = df['publish_date_approx'].dt.day_name()
    return df


