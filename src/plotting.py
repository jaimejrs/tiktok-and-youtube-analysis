# src/plotting.py

import streamlit as st
import plotly.express as px

def plotar_grafico_linha(df, x_col, y_col, agg_func, titulo, **kwargs):
    """Função reutilizável para criar gráficos de linha agregados."""
    df_agg = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
    fig = px.line(df_agg, x=x_col, y=y_col, markers=True, title=titulo, **kwargs)
    st.plotly_chart(fig, use_container_width=True)

def plotar_grafico_barra(df, x_col, y_col, titulo, **kwargs):
    """Função reutilizável para criar gráficos de barra."""
    fig = px.bar(df, x=x_col, y=y_col, title=titulo, **kwargs)
    st.plotly_chart(fig, use_container_width=True)
