# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 12:37:26 2025

@author: rodrigo
"""

# -*- coding: utf-8 -*-
"""
Dashboard Interativo de Análise de Atrasos - iFood
Criado com base na análise exploratória de dados.

@author: rodrigo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Análise de Atrasos iFood",
    page_icon="🍔",
    layout="wide"
)

# --- FUNÇÃO DE CARREGAMENTO DOS DADOS (COM CACHE) ---
# O cache acelera o carregamento do dashboard, pois os dados só são lidos do disco uma vez.
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('C:/Users/rodrigo/Downloads/ds/IfoodCase_tratado.csv')
        
        # Prepara as colunas de tempo para cálculo de média (ignora zeros)
        df['atraso_pedido_min_real'] = df['atraso_pedido_min'].replace(0, np.nan)
        df['atraso_restaurante_min_real'] = df['atraso_restaurante_min'].replace(0, np.nan)
        df['atraso_entregador_min_real'] = df['atraso_entregador_min'].replace(0, np.nan)
        
        # Feature de Período do Dia
        def categoriza_periodo(hora):
            if 5 <= hora < 12: return 'Manhã'
            elif 12 <= hora < 18: return 'Tarde (Almoço)'
            else: return 'Noite (Jantar)'
        df['periodo_dia'] = df['hora_dia'].apply(categoriza_periodo)
        
        return df
    except FileNotFoundError:
        st.error("Arquivo 'IfoodCase_tratado.csv' não encontrado. Por favor, coloque-o na mesma pasta do script.")
        return None

# Carrega os dados usando a função com cache
df_completo = carregar_dados()

if df_completo is not None:
    # --- BARRA LATERAL DE FILTROS ---
    st.sidebar.header("Filtros Interativos")

    # Filtro por Modal
    modal_selecionado = st.sidebar.multiselect(
        'Selecione o Modal:',
        options=df_completo['modal'].unique(),
        default=df_completo['modal'].unique()
    )

    # Filtro por Dia da Semana
    dia_semana_selecionado = st.sidebar.multiselect(
        'Selecione o Dia da Semana:',
        options=df_completo['dia_semana'].unique(),
        default=df_completo['dia_semana'].unique()
    )

    # Filtro por Período do Dia
    periodo_dia_selecionado = st.sidebar.multiselect(
        'Selecione o Período do Dia:',
        options=df_completo['periodo_dia'].unique(),
        default=df_completo['periodo_dia'].unique()
    )

    # Aplicando os filtros ao DataFrame
    df_filtrado = df_completo[
        df_completo['modal'].isin(modal_selecionado) &
        df_completo['dia_semana'].isin(dia_semana_selecionado) &
        df_completo['periodo_dia'].isin(periodo_dia_selecionado)
    ]


    # --- TÍTULO PRINCIPAL ---
    st.title("🚀 Dashboard de Análise de Atrasos - iFood")
    st.markdown("Análise interativa dos principais gargalos na operação de entrega.")


    # --- KPIs PRINCIPAIS ---
    total_pedidos = len(df_filtrado)
    total_atrasados = df_filtrado['flag_atraso_pedido'].sum()
    taxa_atraso = (total_atrasados / total_pedidos) * 100 if total_pedidos > 0 else 0
    atraso_medio = df_filtrado['atraso_pedido_min_real'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pedidos", f"{total_pedidos:,}")
    col2.metric("Taxa de Atraso Geral", f"{taxa_atraso:.2f}%")
    col3.metric("Atraso Médio (dos Atrasados)", f"{atraso_medio:.2f} min")


    # --- ANÁLISE DA CAUSA RAIZ ---
    st.markdown("---")
    st.header("Análise da Causa Raiz: Restaurante vs. Entregador")

    contagem_rest = df_filtrado['flag_atraso_restaurante'].sum()
    contagem_entr = df_filtrado['flag_atraso_entregador'].sum()
    dados_comparacao = {
        'Fonte do Atraso': ['Restaurante', 'Entregador'],
        'Contagem de Atrasos': [contagem_rest, contagem_entr]
    }
    df_comparacao = pd.DataFrame(dados_comparacao)

    fig_causa_raiz = px.bar(df_comparacao, x='Fonte do Atraso', y='Contagem de Atrasos',
                            color='Fonte do Atraso', text_auto=True,
                            title='Contagem de Atrasos por Fonte Principal')
    st.plotly_chart(fig_causa_raiz, use_container_width=True)

    
    # --- ANÁLISES DETALHADAS ---
    st.markdown("---")
    st.header("Análises Detalhadas por Dimensão")

    col_dia, col_periodo = st.columns(2)

    with col_dia:
        # Atraso médio por dia da semana
        analise_dia = df_filtrado.groupby('dia_semana')['atraso_pedido_min_real'].mean().round(2).reset_index()
        fig_dia = px.bar(analise_dia, x='dia_semana', y='atraso_pedido_min_real',
                         title='Atraso Médio por Dia da Semana', text_auto=True,
                         labels={'dia_semana': '', 'atraso_pedido_min_real': 'Atraso Médio (min)'})
        st.plotly_chart(fig_dia, use_container_width=True)

    with col_periodo:
        # Atraso médio por período do dia
        analise_periodo = df_filtrado.groupby('periodo_dia')['atraso_pedido_min_real'].mean().round(2).reset_index()
        fig_periodo = px.bar(analise_periodo, x='periodo_dia', y='atraso_pedido_min_real',
                             title='Atraso Médio por Período do Dia', text_auto=True,
                             labels={'periodo_dia': '', 'atraso_pedido_min_real': 'Atraso Médio (min)'})
        st.plotly_chart(fig_periodo, use_container_width=True)

        
    # --- ANÁLISE DOS RESTAURANTES "VILÕES" ---
    st.markdown("---")
    st.header("Análise de Performance dos Restaurantes")
    
    with st.expander("Clique para ver os Top 10 Restaurantes com mais problemas"):
        col_top_qtd, col_top_tempo = st.columns(2)

        with col_top_qtd:
            # Top 10 por quantidade de atrasos
            top_10_qtd = df_filtrado[df_filtrado['flag_atraso_restaurante'] == True]['id_loja'].value_counts().head(10).reset_index()
            top_10_qtd.columns = ['id_loja', 'contagem_atrasos']
            st.write("**Top 10 por Nº de Atrasos (Causados pelo Restaurante)**")
            st.dataframe(top_10_qtd)

        with col_top_tempo:
            # Top 10 por tempo médio de atraso
            top_10_tempo = df_filtrado.groupby('id_loja')['atraso_restaurante_min_real'].mean().sort_values(ascending=False).head(10).round(2).reset_index()
            st.write("**Top 10 por Pior Média de Atraso (Causado pelo Restaurante)**")
            st.dataframe(top_10_tempo)


else:
    st.warning("Não foi possível carregar os dados. O dashboard não pode ser exibido.")