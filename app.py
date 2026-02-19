import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Fraud Sentinel Pro", layout="wide", page_icon="🛡️")

# Estilização CSS personalizada
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    div[data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px rgba(0,242,255,0.2); }
    .stMetric { background-color: #12141a; padding: 15px; border-radius: 10px; border: 1px solid #2d333b; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO E TRATAMENTO DE DADOS
@st.cache_data
def load_data():
    file_name = 'data.csv'
    df = pd.read_csv(file_name)
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], dayfirst=True, errors='coerce')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dob'] = pd.to_datetime(df['dob'], dayfirst=True, errors='coerce')
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    
    def haversine(lat1, lon1, lat2, lon2):
        r = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * r * np.arcsin(np.sqrt(a))

    df['dist_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    df['avg_cat_amt'] = df.groupby('category')['amt'].transform('mean')
    df['std_cat_amt'] = df.groupby('category')['amt'].transform('std')
    df['z_score_amt'] = (df['amt'] - df['avg_cat_amt']) / (df['std_cat_amt'] + 1e-9)
    df['is_value_anomaly'] = df['z_score_amt'] > 2
    
    dist_mean = df['dist_km'].mean()
    dist_std = df['dist_km'].std()
    df['is_dist_anomaly'] = df['dist_km'] > (dist_mean + 2 * dist_std)
    
    return df

df = load_data()

# 3. SIDEBAR
st.sidebar.title("🛡️ Fraud Sentinel Pro")
st.sidebar.markdown("---")
categorias = st.sidebar.multiselect("Categorias", df['category'].unique(), default=df['category'].unique())
anomalias_apenas = st.sidebar.checkbox("Exibir apenas Alertas")

df_filtered = df[df['category'].isin(categorias)]
if anomalias_apenas:
    df_filtered = df_filtered[(df_filtered['is_value_anomaly']) | (df_filtered['is_dist_anomaly'])]

# 4. CABEÇALHO
st.title("Fraud Monitoring & Advanced Analytics")
st.caption(f"Análise de Risco Geossocial e Estatístico | Desenvolvido por Alexandre C. Passos")

# 5. KPIS
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Volume Filtrado", f"R$ {df_filtered['amt'].sum():,.0f}")
with m2:
    fraud_rate = (df_filtered['is_fraud'].mean() * 100)
    st.metric("Taxa de Fraude", f"{fraud_rate:.2f}%", delta="-0.15%", delta_color="inverse")
with m3:
    st.metric("Distância Média", f"{df_filtered['dist_km'].mean():.1f} km")
with m4:
    st.metric("Anomalias Gasto", int(df_filtered['is_value_anomaly'].sum()), delta="Z-Score > 2", delta_color="inverse")
with m5:
    st.metric("Alertas Distância", int(df_filtered['is_dist_anomaly'].sum()), delta="Outliers Geo", delta_color="inverse")

st.divider()

# 6. VISUAIS PRINCIPAIS
col_map, col_stats = st.columns([2, 1])

with col_map:
    st.subheader("📍 Mapeamento Geográfico de Risco")
    fig_map = px.scatter_mapbox(df_filtered, lat="lat", lon="long", color="is_fraud", 
                                size="amt", color_continuous_scale=["#00f2ff", "#ff3131"],
                                mapbox_style="carto-darkmatter", zoom=3, height=450)
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with col_stats:
    st.subheader("📊 Distribuição Z-Score")
    fig_hist = px.histogram(df_filtered, x="z_score_amt", color="is_fraud",
                            nbins=25, color_discrete_sequence=["#00f2ff", "#ff3131"])
    fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=450)
    st.plotly_chart(fig_hist, use_container_width=True)

# 7. MATRIZ DE RISCO (HEATMAP)
st.divider()
st.subheader("🕒 Matriz de Risco: Hora do Dia vs. Perfil Demográfico")

bins = [0, 25, 40, 60, 100]
labels = ['Jovens (0-25)', 'Adultos (26-40)', 'Sêniors (41-60)', 'Idosos (60+)']
df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=bins, labels=labels)

heatmap_data = df_filtered.pivot_table(index='age_group', columns='hour', values='is_fraud', aggfunc='sum').fillna(0)

fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Hora do Dia", y="Perfil", color="Fraudes"),
                        x=heatmap_data.columns, y=heatmap_data.index,
                        color_continuous_scale='Reds', aspect="auto")
fig_heatmap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
st.plotly_chart(fig_heatmap, use_container_width=True)

# 8. TABELA DE AUDITORIA (VERSÃO COMPATÍVEL)
st.divider()
st.subheader("🕵️ Tabela de Investigação (Top Alertas)")
audit_df = df_filtered[['trans_date_trans_time', 'category', 'amt', 'dist_km', 'z_score_amt', 'is_fraud']]
audit_df = audit_df.sort_values(by='z_score_amt', ascending=False).head(20)

# Formatação direta via st.column_config (Mais robusto e moderno)
st.dataframe(
    audit_df,
    column_config={
        "amt": st.column_config.NumberColumn("Valor", format="R$ %.2f"),
        "dist_km": st.column_config.NumberColumn("Distância", format="%.2f km"),
        "z_score_amt": st.column_config.NumberColumn("Z-Score", format="%.2f"),
        "is_fraud": st.column_config.CheckboxColumn("Fraude Conf.")
    },
    use_container_width=True,
    hide_index=True
)
