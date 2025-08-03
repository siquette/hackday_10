# -*- coding: utf-8 -*-
"""
Dashboard Estratégico de Análise de Atrasos - iFood (Versão Final com Regressão)

@author: rodrigo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Estratégico iFood",
    page_icon="🍔",
    layout="wide"
)

# --- FUNÇÃO DE CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('IfoodCase_tratado.csv')
        
        # Conversões de tipo
        df['hora_data_pedido'] = pd.to_datetime(df['hora_data_pedido'], errors='coerce')
        df['data_pedido'] = pd.to_datetime(df['data_pedido'], errors='coerce')
        df.dropna(subset=['hora_data_pedido', 'data_pedido'], inplace=True)
        
        # Colunas para cálculo de média (ignora zeros)
        df['atraso_pedido_min_real'] = df['atraso_pedido_min'].replace(0, np.nan)
        df['atraso_restaurante_min_real'] = df['atraso_restaurante_min'].replace(0, np.nan)
        
        # Feature de Período do Dia
        def categoriza_periodo(hora):
            if 5 <= hora < 12: return 'Manhã'
            elif 12 <= hora < 18: return 'Tarde (Almoço)'
            else: return 'Noite (Jantar)'
        df['periodo_dia'] = df['hora_dia'].apply(categoriza_periodo)
        
        # Feature de Faixa de Distância
        bins = [0, 3, 7, 15, df['distancia_restaurante_cliente_km'].max() + 1]
        labels = ['Curta (0-3 km)', 'Média (3-7 km)', 'Longa (7-15 km)', 'Muito Longa (15+ km)']
        df['faixa_distancia'] = pd.cut(df['distancia_restaurante_cliente_km'], bins=bins, labels=labels, right=False)
        
        return df
    except FileNotFoundError:
        st.error("Arquivo 'IfoodCase_tratado.csv' não encontrado.")
        return None

# --- FUNÇÕES DE TREINAMENTO DOS MODELOS (COM CACHE) ---
@st.cache_resource
def treinar_modelo_linear(df):
    features = ['tempo_preparo_real_min', 'distancia_restaurante_cliente_km', 'hora_dia', 'dia_semana', 'modal']
    alvo_linear = 'atraso_pedido_min'
    df_reg_linear = df[df['flag_atraso_pedido'] == True].copy()
    
    X = df_reg_linear[features]
    y = df_reg_linear[alvo_linear]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    colunas_numericas = X.select_dtypes(include=np.number).columns
    colunas_categoricas = X.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', colunas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
    ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    return pipeline, y_test, y_pred

@st.cache_resource
def treinar_modelo_logistico(df):
    features = ['tempo_preparo_real_min', 'distancia_restaurante_cliente_km', 'hora_dia', 'dia_semana', 'modal']
    alvo_logistica = 'flag_atraso_pedido'
    
    X = df[features]
    y = df[alvo_logistica]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    colunas_numericas = X.select_dtypes(include=np.number).columns
    colunas_categoricas = X.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', colunas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
    ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    return pipeline, y_test, y_pred

# --- FUNÇÕES PARA RENDERIZAR CADA PÁGINA ---

def pagina_apresentacao():
    st.title("Desafio Analítico iFood – Delivery Optimization Challenge")
    st.image("https://logodownload.org/wp-content/uploads/2017/05/ifood-logo-0.png", width=200)
    
    st.header("Contexto do Negócio")
    st.markdown("""
    O iFood realiza milhões de entregas por mês. Com essa operação massiva, pequenos gargalos logísticos se tornam problemas escaláveis que impactam diretamente a experiência do cliente e a reputação da marca.
    """)
    st.warning("**PROBLEMA DE NEGÓCIO:** O time de operações do iFood sofre com a **baixa visibilidade operacional** para responder rapidamente a perguntas críticas.")
    st.info("""**PERGUNTAS-CHAVE:**
    - Quais restaurantes estão impactando negativamente a logística?
    - Existe algum padrão de atraso em horários ou dias específicos?
    - O atraso vem do restaurante ou do entregador?
    """)

def pagina_visao_geral(df_filtrado):
    st.title("📈 Visão Geral da Operação")
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        total_pedidos = len(df_filtrado)
        taxa_atraso_geral = (df_filtrado['flag_atraso_pedido'].sum() / total_pedidos) * 100
        media_atraso = df_filtrado['atraso_pedido_min_real'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Pedidos", f"{total_pedidos:,}")
        col2.metric("Taxa de Atraso Geral", f"{taxa_atraso_geral:.2f}%")
        col3.metric("Atraso Médio (dos Atrasados)", f"{media_atraso:.2f} min")
        
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

def pagina_analise_temporal(df_filtrado):
    st.title("📈 Análise Temporal (Tendências)")
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        st.subheader("KPIs Diários")
        df_resampled = df_filtrado.set_index('hora_data_pedido').resample('D').agg({'numero_pedido': 'count', 'atraso_pedido_min_real': 'mean', 'flag_atraso_pedido': 'sum'})
        df_resampled['taxa_atraso'] = (df_resampled['flag_atraso_pedido'] / df_resampled['numero_pedido'] * 100).round(2)
        fig_temporal = px.line(df_resampled, y=['numero_pedido', 'taxa_atraso', 'atraso_pedido_min_real'], title='Evolução Diária dos KPIs', facet_row="variable", labels={'value': 'Valor', 'hora_data_pedido': 'Data'})
        fig_temporal.update_yaxes(matches=None)
        st.plotly_chart(fig_temporal, use_container_width=True)
        st.subheader("Mapa de Calor de Atrasos (Dia da Semana vs. Hora)")
        heatmap_data = df_filtrado.pivot_table(index='hora_dia', columns='dia_semana', values='flag_atraso_pedido', aggfunc='mean')
        heatmap_data = heatmap_data.reindex(columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig_heatmap = px.imshow(heatmap_data, text_auto=".1%", color_continuous_scale='Reds', title='Taxa de Atraso Média por Hora e Dia da Semana')
        st.plotly_chart(fig_heatmap, use_container_width=True)

def pagina_analise_distancia(df_filtrado):
    st.title("🗺️ Análise por Distância")
    if df_filtrado.empty or df_filtrado['faixa_distancia'].dropna().empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados ou faixas de distância.")
    else:
        st.subheader("Causa do Atraso por Faixa de Distância")
        analise_dist_qtd = df_filtrado.groupby('faixa_distancia', observed=True).agg(atrasos_restaurante=('flag_atraso_restaurante', 'sum'), atrasos_entregador=('flag_atraso_entregador', 'sum')).reset_index()
        fig_dist_causa = px.bar(analise_dist_qtd, x='faixa_distancia', y=['atrasos_restaurante', 'atrasos_entregador'], barmode='group', title='Quantidade de Atrasos (Restaurante vs. Entregador) por Distância')
        st.plotly_chart(fig_dist_causa, use_container_width=True)
        st.subheader("Uso de Modais por Faixa de Distância")
        crosstab_modal = pd.crosstab(df_filtrado['faixa_distancia'], df_filtrado['modal'], normalize='index') * 100
        fig_dist_modal = px.bar(crosstab_modal, title='Proporção de Uso de Modais por Distância', text_auto='.1f' + '%')
        st.plotly_chart(fig_dist_modal, use_container_width=True)

def pagina_performance_lojas(df_filtrado):
    st.title("🏢 Análise de Performance das Lojas")
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        st.subheader("Ranking de Lojas por Score de Performance")
        lojas_agg = df_filtrado.groupby('id_loja').agg(total_pedidos=('numero_pedido', 'count'), total_atrasos_rest=('flag_atraso_restaurante', 'sum'), media_tempo_preparo=('tempo_preparo_real_min', 'mean')).reset_index()
        lojas_agg['taxa_atraso_restaurante'] = (lojas_agg['total_atrasos_rest'] / lojas_agg['total_pedidos'])
        lojas_agg = lojas_agg[lojas_agg['media_tempo_preparo'] > 0]
        lojas_agg['ScoreLoja'] = (1 - lojas_agg['taxa_atraso_restaurante']) * (10 / lojas_agg['media_tempo_preparo'])
        lojas_ranqueadas = lojas_agg.sort_values('ScoreLoja', ascending=True)
        st.dataframe(lojas_ranqueadas[['id_loja', 'ScoreLoja', 'taxa_atraso_restaurante', 'media_tempo_preparo', 'total_pedidos']].round(2))
        st.subheader("Análise Visual de Performance das Lojas")
        fig_scatter_lojas = px.scatter(lojas_ranqueadas, x='media_tempo_preparo', y='taxa_atraso_restaurante', size='total_pedidos', color='ScoreLoja', hover_name='id_loja', color_continuous_scale='RdYlGn_r', title="Performance das Lojas (Tempo de Preparo vs. Taxa de Atraso)")
        st.plotly_chart(fig_scatter_lojas, use_container_width=True)

def pagina_analise_outliers(df_filtrado):
    st.title("📊 Análise de Outliers")
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        df_pedido_atrasado = df_filtrado[df_filtrado['flag_atraso_pedido'] == True]
        if df_pedido_atrasado.empty:
            st.info("Não há pedidos atrasados para os filtros selecionados.")
        else:
            st.subheader(f"Análise de Outliers para Pedidos Atrasados ({len(df_pedido_atrasado)} ocorrências)")
            colunas_pedido = ['atraso_pedido_min', 'distancia_restaurante_cliente_km', 'tempo_preparo_real_min']
            for col in colunas_pedido:
                fig = px.box(df_pedido_atrasado, y=col, title=f'Boxplot para Atraso Geral: {col}')
                st.plotly_chart(fig, use_container_width=True)

def pagina_correlacao(df_filtrado):
    st.title("🔗 Análise de Correlação")
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        colunas_importantes = ['flag_atraso_pedido', 'atraso_pedido_min', 'flag_atraso_restaurante', 'atraso_restaurante_min', 'flag_atraso_entregador', 'atraso_entregador_min', 'tempo_preparo_real_min', 'tempo_ida_restaurante_min', 'tempo_percurso_entrega_min', 'distancia_restaurante_cliente_km', 'hora_dia']
        df_corr = df_filtrado[colunas_importantes].copy()
        for col in df_corr.select_dtypes(include=['bool']).columns:
            df_corr[col] = df_corr[col].astype(int)
        correlation_matrix = df_corr.corr()
        st.subheader("Heatmap de Correlação das Variáveis Mais Importantes")
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu', title='Heatmap de Correlação Focado nas Causas do Atraso')
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

def pagina_conclusoes():
    st.title("💡 Conclusões e Recomendações")
    st.success("**Insight 1: O gargalo da operação está nos restaurantes.** A duração do atraso do restaurante e o tempo de preparo são as variáveis com maior correlação com o atraso total. **Recomendação:** Focar em programas de eficiência para os restaurantes da aba 'Performance das Lojas'.")
    st.info("**Insight 2: A distância não é um fator crítico para os atrasos.** A correlação é muito baixa. **Recomendação:** Garantir que o modal correto seja alocado (motos para distâncias longas), em vez de focar em limitar o raio de entrega.")
    st.warning("**Insight 3: Picos de atraso são previsíveis.** A taxa de atraso aumenta nos horários de pico (noite) e nos fins de semana (Sexta e Sábado). **Recomendação:** Implementar uma alocação dinâmica de recursos e incentivos nesses horários críticos.")

def pagina_regressao(df_completo):
    st.title("🤖 Modelagem Preditiva (Regressão)")
    st.markdown("Utilizando Machine Learning para prever a ocorrência e a duração dos atrasos.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Modelo 1: O Pedido VAI Atrasar?")
        st.subheader("(Regressão Logística)")
        pipeline_log, y_test_log, y_pred_log = treinar_modelo_logistico(df_completo)
        accuracy = accuracy_score(y_test_log, y_pred_log)
        st.metric("Acurácia do Modelo", f"{accuracy:.2%}")
        cm = confusion_matrix(y_test_log, y_pred_log)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Previsão", y="Real"), x=['Não Atrasou', 'Atrasou'], y=['Não Atrasou', 'Atrasou'], title="Matriz de Confusão")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.header("Modelo 2: QUANTO o Pedido Vai Atrasar?")
        st.subheader("(Regressão Linear)")
        pipeline_lin, y_test_lin, y_pred_lin = treinar_modelo_linear(df_completo)
        r2 = r2_score(y_test_lin, y_pred_lin)
        rmse = np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))
        
        st.metric("Coeficiente de Determinação (R²)", f"{r2:.2f}")
        st.metric("Erro Médio da Previsão (RMSE)", f"{rmse:.2f} minutos")
        
        df_resultados = pd.DataFrame({'Real': y_test_lin, 'Previsto': y_pred_lin})
        fig_scatter = px.scatter(df_resultados, x='Real', y='Previsto', title='Valores Reais vs. Previstos', labels={'Real': 'Atraso Real (Min)', 'Previsto': 'Atraso Previsto (Min)'}, opacity=0.5)
        fig_scatter.add_shape(type='line', x0=0, y0=0, x1=df_resultados['Real'].max(), y1=df_resultados['Real'].max(), line=dict(color='red', width=2, dash='dash'))
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- LÓGICA PRINCIPAL DO APP ---
df_completo = carregar_dados()

if df_completo is not None:
    st.sidebar.image("https://logodownload.org/wp-content/uploads/2017/05/ifood-logo-0.png", width=150)
    st.sidebar.title("Navegação")
    
    paginas = ["🏠 Apresentação", "📈 Visão Geral", "📈 Análise Temporal", "🗺️ Análise por Distância", 
               "🏢 Performance das Lojas", "📊 Análise de Outliers", "🔗 Análise de Correlação", 
               "💡 Conclusões", "🤖 Modelagem Preditiva"]
    pagina_selecionada = st.sidebar.radio("Selecione uma Análise:", paginas)
    
    if pagina_selecionada not in ["🏠 Apresentação", "💡 Conclusões", "🤖 Modelagem Preditiva"]:
        st.sidebar.markdown("---")
        st.sidebar.header("Segmentação de Dados")
        modal_selecionado = st.sidebar.multiselect('Modal:', options=df_completo['modal'].unique(), default=df_completo['modal'].unique())
        dia_semana_selecionado = st.sidebar.multiselect('Dia da Semana:', options=df_completo['dia_semana'].unique(), default=df_completo['dia_semana'].unique())
        df_filtrado = df_completo[df_completo['modal'].isin(modal_selecionado) & df_completo['dia_semana'].isin(dia_semana_selecionado)]
    else:
        df_filtrado = df_completo.copy()

    # --- ROTEAMENTO DAS PÁGINAS ---
    if pagina_selecionada == "🏠 Apresentação":
        pagina_apresentacao()
    elif pagina_selecionada == "📈 Visão Geral":
        pagina_visao_geral(df_filtrado)
    elif pagina_selecionada == "📈 Análise Temporal":
        pagina_analise_temporal(df_filtrado)
    elif pagina_selecionada == "🗺️ Análise por Distância":
        pagina_analise_distancia(df_filtrado)
    elif pagina_selecionada == "🏢 Performance das Lojas":
        pagina_performance_lojas(df_filtrado)
    elif pagina_selecionada == "📊 Análise de Outliers":
        pagina_analise_outliers(df_filtrado)
    elif pagina_selecionada == "🔗 Análise de Correlação":
        pagina_correlacao(df_filtrado)
    elif pagina_selecionada == "💡 Conclusões":
        pagina_conclusoes()
    elif pagina_selecionada == "🤖 Modelagem Preditiva":
        pagina_regressao(df_completo)
else:
    st.warning("Não foi possível carregar os dados. O dashboard não pode ser exibido.")