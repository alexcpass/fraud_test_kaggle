import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Fraud Sentinel Pro", layout="wide", page_icon="🛡️")

# Estilização para manter o padrão Premium Dark
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    div[data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.2); }
    .stMetric { background-color: #12141a; padding: 15px; border-radius: 10px; border: 1px solid #2d333b; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Carregando os dados
    df = pd.read_csv('fraudTest_amostra.xlsx - Planilha1.csv')
    
    # 1. Tratamento de Datas e Idade
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    
    # 2. Cálculo de Distância (Haversine)
    def haversine(lat1, lon1, lat2, lon2):
        r = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * r * np.arcsin(np.sqrt(a))

    df['dist_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    # --- CÁLCULOS AVANÇADOS (SOLICITADOS) ---
    
    # 3. Desvio Padrão de Gastos por Categoria (Z-Score)
    # Identifica o quão longe o gasto atual está da média daquela categoria
    df['avg_cat_amt'] = df.groupby('category')['amt'].transform('mean')
    df['std_cat_amt'] = df.groupby('category')['amt'].transform('std')
    df['z_score_amt'] = (df['amt'] - df['avg_cat_amt']) / df['std_cat_amt']
    
    # Flag de Anomalia de Valor (Gasto > 2 desvios padrões da média da categoria)
    df['is_value_anomaly'] = df['z_score_amt'] > 2
    
    # 4. Alerta de Proximidade (Outlier de Distância)
    # Sinaliza transações que ocorrem a uma distância superior a 2 desvios padrões da média global
    dist_mean = df['dist_km'].mean()
    dist_std = df['dist_km'].std()
    df['is_dist_anomaly'] = df['dist_km'] > (dist_mean + 2 * dist_std)
    
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🛡️ Filtros Sentinel")
st.sidebar.markdown("---")
categorias = st.sidebar.multiselect("Categorias", df['category'].unique(), default=df['category'].unique())
anomalias_apenas = st.sidebar.checkbox("Exibir apenas Anomalias (Alertas)")

# Lógica de Filtro
df_filtered = df[df['category'].isin(categorias)]
if anomalias_apenas:
    df_filtered = df_filtered[(df_filtered['is_value_anomaly']) | (df_filtered['is_dist_anomaly'])]

# --- DASHBOARD PRINCIPAL ---
st.title("Fraud Monitoring & Advanced Analytics")
st.caption("Sistema de Detecção de Anomalias Estatísticas | Analista: Alexandre C. Passos")

# --- ROW 1: MÉTRICAS BÁSICAS E AVANÇADAS ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Transacionado", f"R$ {df_filtered['amt'].sum():,.2f}")
with m2:
    fraud_rate = (df_filtered['is_fraud'].mean() * 100)
    st.metric("Taxa de Fraude", f"{fraud_rate:.2f}%")
with m3:
    st.metric("Distância Média", f"{df_filtered['dist_km'].mean():.1f} km")
with m4:
    val_anomalias = df_filtered['is_value_anomaly'].sum()
    st.metric("Anomalias de Gasto", val_anomalias, delta="Acima de 2σ", delta_color="inverse")
with m5:
    dist_anomalias = df_filtered['is_dist_anomaly'].sum()
    st.metric("Alertas de Distância", dist_anomalias, delta="Fora do Raio", delta_color="inverse")

st.markdown("---")

# --- ROW 2: VISUAIS ---
col_map, col_stats = st.columns([2, 1])

with col_map:
    st.subheader("📍 Mapeamento Geográfico e Alertas")
    # No mapa, vamos destacar as anomalias com um tamanho maior
    fig_map = px.scatter_mapbox(df_filtered, lat="lat", lon="long", 
                                color="is_fraud", size="amt",
                                hover_name="merchant", 
                                hover_data=["dist_km", "z_score_amt"],
                                color_continuous_scale=["#00f2ff", "#ff3131"],
                                mapbox_style="carto-darkmatter", zoom=3, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

with col_stats:
    st.subheader("📊 Distribuição de Gasto (Z-Score)")
    # Histograma para mostrar a curva de desvio
    fig_hist = px.histogram(df_filtered, x="z_score_amt", color="is_fraud",
                            nbins=30, barmode="overlay",
                            color_discrete_sequence=["#00f2ff", "#ff3131"])
    fig_hist.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

# --- ROW 3: ANÁLISE DE SEGURANÇA ---
st.subheader("🕵️ Tabela de Investigação (Top Alertas)")
# Criando um dataframe de auditoria ordenado pelas maiores anomalias
audit_df = df_filtered[
    ['trans_date_trans_time', 'category', 'amt', 'dist_km', 'z_score_amt', 'is_value_anomaly', 'is_dist_anomaly']
].sort_values(by='z_score_amt', ascending=False)

st.dataframe(
    audit_df.style.format({'amt': 'R$ {:.2f}', 'dist_km': '{:.1f} km', 'z_score_amt': '{:.2f}'})
    .highlight_between(left=2, right=100, subset=['z_score_amt'], color='#440000') # Destaca Z-Score alto
)