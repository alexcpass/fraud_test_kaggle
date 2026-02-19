import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Fraud Sentinel Pro", layout="wide", page_icon="🛡️")

# Estilização Premium Dark
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    div[data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.2); }
    .stMetric { background-color: #12141a; padding: 15px; border-radius: 10px; border: 1px solid #2d333b; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # NOME DO ARQUIVO ATUALIZADO AQUI
    file_name = 'fraudTest_amostra.csv'
    df = pd.read_csv(file_name)
    
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
    
    # 3. Estatística Avançada: Z-Score de Gastos por Categoria
    df['avg_cat_amt'] = df.groupby('category')['amt'].transform('mean')
    df['std_cat_amt'] = df.groupby('category')['amt'].transform('std')
    df['z_score_amt'] = (df['amt'] - df['avg_cat_amt']) / df['std_cat_amt']
    df['is_value_anomaly'] = df['z_score_amt'] > 2
    
    # 4. Outlier de Distância (> 2 desvios padrões)
    dist_mean = df['dist_km'].mean()
    dist_std = df['dist_km'].std()
    df['is_dist_anomaly'] = df['dist_km'] > (dist_mean + 2 * dist_std)
    
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🛡️ Filtros Sentinel")
categorias = st.sidebar.multiselect("Categorias", df['category'].unique(), default=df['category'].unique())
anomalias_apenas = st.sidebar.checkbox("Exibir apenas Anomalias (Alertas)")

df_filtered = df[df['category'].isin(categorias)]
if anomalias_apenas:
    df_filtered = df_filtered[(df_filtered['is_value_anomaly']) | (df_filtered['is_dist_anomaly'])]

# --- DASHBOARD PRINCIPAL ---
st.title("Fraud Monitoring & Advanced Analytics")
st.caption("Sistema de Detecção de Anomalias Estatísticas | Analista: Alexandre C. Passos")

# --- MÉTRICAS ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Transacionado", f"R$ {df_filtered['amt'].sum():,.2f}")
with m2:
    st.metric("Taxa de Fraude", f"{(df_filtered['is_fraud'].mean() * 100):.2f}%")
with m3:
    st.metric("Distância Média", f"{df_filtered['dist_km'].mean():.1f} km")
with m4:
    st.metric("Anomalias de Gasto", df_filtered['is_value_anomaly'].sum())
with m5:
    st.metric("Alertas de Distância", df_filtered['is_dist_anomaly'].sum())

st.divider()

# --- VISUAIS ---
col_map, col_stats = st.columns([2, 1])

with col_map:
    st.subheader("📍 Mapeamento Geográfico")
    fig_map = px.scatter_mapbox(df_filtered, lat="lat", lon="long", color="is_fraud", 
                                size="amt", color_continuous_scale=["#00f2ff", "#ff3131"],
                                mapbox_style="carto-darkmatter", zoom=3, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

with col_stats:
    st.subheader("📊 Curva de Desvio (Z-Score)")
    fig_hist = px.histogram(df_filtered, x="z_score_amt", color="is_fraud",
                            nbins=30, color_discrete_sequence=["#00f2ff", "#ff3131"])
    fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

# --- TABELA ---
st.subheader("🕵️ Detalhes para Auditoria")
st.dataframe(df_filtered[['trans_date_trans_time', 'category', 'amt', 'dist_km', 'z_score_amt']].sort_values(by='z_score_amt', ascending=False))
