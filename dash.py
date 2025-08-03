# -*- coding: utf-8 -*-
"""
Dashboard Multi-Página de Análise de Atrasos - iFood (Versão Final Corrigida)

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

# --- FUNÇÃO DE CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('IfoodCase_tratado.csv')
        
        # --- INÍCIO DA CORREÇÃO ---
        # Adicionamos errors='coerce' para lidar com formatos de data inválidos
        df['hora_data_pedido'] = pd.to_datetime(df['hora_data_pedido'], errors='coerce')
        df['data_pedido'] = pd.to_datetime(df['data_pedido'], errors='coerce')
        
        # Removemos qualquer linha onde a data não pôde ser convertida (ficou NaT)
        df.dropna(subset=['hora_data_pedido', 'data_pedido'], inplace=True)
        # --- FIM DA CORREÇÃO ---
        
        # Colunas para cálculo de média (ignora zeros)
        df['atraso_pedido_min_real'] = df['atraso_pedido_min'].replace(0, np.nan)
        df['atraso_restaurante_min_real'] = df['atraso_restaurante_min'].replace(0, np.nan)
        
        # Feature de Período do Dia
        def categoriza_periodo(hora):
            if 5 <= hora < 12: return 'Manhã'
            elif 12 <= hora < 18: return 'Tarde (Almoço)'
            else: return 'Noite (Jantar)'
        df['periodo_dia'] = df['hora_dia'].apply(categoriza_periodo)
        
        return df
    except FileNotFoundError:
        st.error("Arquivo 'IfoodCase_tratado.csv' não encontrado.")
        return None

# --- FUNÇÕES PARA RENDERIZAR CADA PÁGINA ---

def pagina_visao_geral(df_filtrado):
    st.title("🏠 Visão Geral da Operação")
    st.markdown("KPIs e métricas macro dos pedidos.")

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        total_pedidos = len(df_filtrado)
        taxa_atraso_geral = (df_filtrado['flag_atraso_pedido'].sum() / total_pedidos) * 100
        media_atraso = df_filtrado['atraso_pedido_min_real'].mean()
        dist_media = df_filtrado['distancia_restaurante_cliente_km'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Pedidos", f"{total_pedidos:,}")
        col2.metric("Taxa de Atraso Geral", f"{taxa_atraso_geral:.2f}%")
        col3.metric("Atraso Médio (dos Atrasados)", f"{media_atraso:.2f} min")
        col4.metric("Distância Média", f"{dist_media:.2f} km")
        
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Divisão da Causa do Atraso")
            causas = df_filtrado[['flag_atraso_restaurante', 'flag_atraso_entregador']].sum().reset_index()
            causas.columns = ['Causa', 'Contagem']
            causas['Causa'] = causas['Causa'].replace({'flag_atraso_restaurante': 'Restaurante', 'flag_atraso_entregador': 'Entregador'})
            fig_pie = px.pie(causas, names='Causa', values='Contagem', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            st.subheader("Taxa de Atraso por Dia da Semana")
            taxa_atraso_dia = (df_filtrado.groupby('dia_semana')['flag_atraso_pedido'].sum() / df_filtrado.groupby('dia_semana')['numero_pedido'].count() * 100).round(2).reset_index(name='taxa_atraso')
            fig_bar_dia = px.bar(taxa_atraso_dia, x='dia_semana', y='taxa_atraso', text_auto=True)
            fig_bar_dia.update_layout(yaxis_title="Taxa de Atraso (%)", xaxis_title="")
            st.plotly_chart(fig_bar_dia, use_container_width=True)

def pagina_etapas_entrega(df_filtrado):
    st.title("🚚 Análise das Etapas da Entrega")
    st.markdown("Análise detalhada dos tempos em cada fase do pedido.")
    
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        st.subheader("Tempo Médio por Etapa e Modal")
        etapas = df_filtrado.groupby('modal')[['tempo_ida_restaurante_min', 'tempo_preparo_real_min', 'tempo_percurso_entrega_min']].mean().reset_index()
        fig_etapas = px.bar(etapas, x='modal', y=['tempo_ida_restaurante_min', 'tempo_preparo_real_min', 'tempo_percurso_entrega_min'],
                            title="Tempo Médio por Etapa (min)", text_auto='.2f')
        st.plotly_chart(fig_etapas, use_container_width=True)
        
        st.subheader("Taxa de Atraso do Entregador por Modal")
        taxa_atraso_entregador = (df_filtrado.groupby('modal')['flag_atraso_entregador'].sum() / df_filtrado.groupby('modal')['numero_pedido'].count() * 100).round(2).reset_index(name='taxa_atraso')
        fig_bar_entregador = px.bar(taxa_atraso_entregador, x='modal', y='taxa_atraso', text_auto=True)
        fig_bar_entregador.update_layout(yaxis_title="Taxa de Atraso do Entregador (%)", xaxis_title="Modal")
        st.plotly_chart(fig_bar_entregador, use_container_width=True)

def pagina_performance_lojas(df_filtrado):
    st.title("🏢 Análise de Performance das Lojas")
    st.markdown("Ranking e análise visual para identificar restaurantes com baixa performance.")

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        st.subheader("Ranking de Lojas por Score de Performance")
        lojas_agg = df_filtrado.groupby('id_loja').agg(
            total_pedidos=('numero_pedido', 'count'),
            total_atrasos_rest=('flag_atraso_restaurante', 'sum'),
            media_tempo_preparo=('tempo_preparo_real_min', 'mean')
        ).reset_index()
        lojas_agg['taxa_atraso_restaurante'] = (lojas_agg['total_atrasos_rest'] / lojas_agg['total_pedidos'])
        lojas_agg = lojas_agg[lojas_agg['media_tempo_preparo'] > 0]
        lojas_agg['ScoreLoja'] = (1 - lojas_agg['taxa_atraso_restaurante']) * (10 / lojas_agg['media_tempo_preparo'])
        
        lojas_ranqueadas = lojas_agg.sort_values('ScoreLoja', ascending=True)
        st.dataframe(lojas_ranqueadas[['id_loja', 'ScoreLoja', 'taxa_atraso_restaurante', 'media_tempo_preparo', 'total_pedidos']].round(2))

        st.subheader("Análise Visual de Performance das Lojas")
        fig_scatter_lojas = px.scatter(
            lojas_ranqueadas, x='media_tempo_preparo', y='taxa_atraso_restaurante',
            size='total_pedidos', color='ScoreLoja', hover_name='id_loja',
            color_continuous_scale='RdYlGn_r', title="Performance das Lojas (Tempo de Preparo vs. Taxa de Atraso)"
        )
        st.plotly_chart(fig_scatter_lojas, use_container_width=True)

def pagina_analise_outliers(df_filtrado):
    st.title("📊 Análise de Outliers")
    st.markdown("Investigação dos casos extremos, separando por causa do atraso.")

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        df_pedido_atrasado = df_filtrado[df_filtrado['flag_atraso_pedido'] == True]
        st.subheader(f"Análise de Outliers para Pedidos Atrasados ({len(df_pedido_atrasado)} ocorrências)")
        colunas_pedido = ['atraso_pedido_min', 'distancia_restaurante_cliente_km', 'tempo_preparo_real_min']
        for col in colunas_pedido:
            fig = px.box(df_pedido_atrasado, y=col, title=f'Boxplot para Atraso Geral: {col}')
            st.plotly_chart(fig, use_container_width=True)
        
        fig_scatter_pedido = px.scatter(df_pedido_atrasado, x='distancia_restaurante_cliente_km', y='atraso_pedido_min', title='Dispersão: Distância vs. Atraso Total do Pedido')
        st.plotly_chart(fig_scatter_pedido, use_container_width=True)

def pagina_correlacao(df_filtrado):
    st.title("🔗 Análise de Correlação")
    st.markdown("Investigação numérica das relações entre as variáveis que podem explicar os atrasos.")

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        colunas_importantes = [
            'flag_atraso_pedido', 'atraso_pedido_min', 'flag_atraso_restaurante',
            'atraso_restaurante_min', 'flag_atraso_entregador', 'atraso_entregador_min',
            'tempo_preparo_real_min', 'tempo_ida_restaurante_min', 'tempo_percurso_entrega_min',
            'distancia_restaurante_cliente_km', 'hora_dia'
        ]
        df_corr = df_filtrado[colunas_importantes].copy()
        
        for col in df_corr.select_dtypes(include=['bool']).columns:
            df_corr[col] = df_corr[col].astype(int)
            
        correlation_matrix = df_corr.corr()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlação com a DURAÇÃO do Atraso")
            corr_duracao = correlation_matrix[['atraso_pedido_min']].sort_values(by='atraso_pedido_min', ascending=False)
            st.dataframe(corr_duracao)
        with col2:
            st.subheader("Correlação com a OCORRÊNCIA do Atraso")
            corr_ocorrencia = correlation_matrix[['flag_atraso_pedido']].sort_values(by='flag_atraso_pedido', ascending=False)
            st.dataframe(corr_ocorrencia)

        st.subheader("Heatmap de Correlação das Variáveis Mais Importantes")
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu', title='Heatmap de Correlação Focado nas Causas do Atraso')
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

# --- LÓGICA PRINCIPAL DO APP ---
df_completo = carregar_dados()

if df_completo is not None:
    st.sidebar.image("https://logodownload.org/wp-content/uploads/2017/05/ifood-logo-0.png", width=150)
    st.sidebar.title("Navegação")
    
    pagina_selecionada = st.sidebar.radio(
        "Selecione uma Análise:",
        ["🏠 Visão Geral", "🚚 Etapas da Entrega", "🏢 Performance das Lojas", "📊 Análise de Outliers", "🔗 Análise de Correlação"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Segmentação de Dados")
    
    modal_selecionado = st.sidebar.multiselect('Modal:', options=df_completo['modal'].unique(), default=df_completo['modal'].unique())
    dia_semana_selecionado = st.sidebar.multiselect('Dia da Semana:', options=df_completo['dia_semana'].unique(), default=df_completo['dia_semana'].unique())
    
    df_filtrado = df_completo[
        df_completo['modal'].isin(modal_selecionado) &
        df_completo['dia_semana'].isin(dia_semana_selecionado)
    ]

    # --- ROTEAMENTO DAS PÁGINAS ---
    if pagina_selecionada == "🏠 Visão Geral":
        pagina_visao_geral(df_filtrado)
    elif pagina_selecionada == "🚚 Etapas da Entrega":
        pagina_etapas_entrega(df_filtrado)
    elif pagina_selecionada == "🏢 Performance das Lojas":
        pagina_performance_lojas(df_filtrado)
    elif pagina_selecionada == "📊 Análise de Outliers":
        pagina_analise_outliers(df_filtrado)
    elif pagina_selecionada == "🔗 Análise de Correlação":
        pagina_correlacao(df_filtrado)
else:
    st.warning("Não foi possível carregar os dados. O dashboard não pode ser exibido.")