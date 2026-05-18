import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
df=pd.read_csv('new_data.csv')
st.set_page_config(
    page_title="Algérie Télécom - Dashboard PFE",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <div style="background-color:#0056b3; padding:20px; border-radius:12px; margin-bottom:25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color:white; text-align:center; margin:0; font-family:sans-serif; font-size:30px; font-weight:bold;">Algérie Télécom — Panneau de commande interactif des pannes</h1>
        <p style="color:#e6f2ff; text-align:center; margin:8px 0 0 0; font-size:15px;">Analyse Visualisation & Prédiction — Data Science Graduation Project (PFE)</p>
    </div>
""", unsafe_allow_html=True)

df_raw = None
csv_filename = 'new_data.csv'

if os.path.exists(csv_filename):
    try:
        df_raw = pd.read_csv(csv_filename, low_memory=False)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la lecture du fichier automatique : {e}")

if df_raw is None:
    st.info("👋  ! Veuillez sélectionner ou glisser votre fichier de données (new_data.csv)  :")
    uploaded_file = st.file_uploader("Choisissez le fichier CSV pour Algeria Telecom", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, low_memory=False)
    else:
        st.stop()

@st.cache_data
def preprocess_clean_data(df):
    df.columns = df.columns.str.strip().str.upper()
    
    essential_cols = {'PROVINCE': 'Inconnu', 'DISTRICTINS': 'Inconnu', 'TYPE_DERG': 'Autre', 'TT_STATUS': 'Closed'}
    for col, def_val in essential_cols.items():
        if col not in df.columns:
            df[col] = def_val

    if 'SUBMIT_DATE' in df.columns:
        df['SUBMIT_DATE_CLEAN'] = pd.to_datetime(df['SUBMIT_DATE'], errors='coerce')
        df['SUBMIT_DATE_CLEAN'] = df['SUBMIT_DATE_CLEAN'].fillna(pd.to_datetime('2026-05-18'))
    else:
        df['SUBMIT_DATE_CLEAN'] = pd.to_datetime('2026-05-18')

    if 'CLOSE_DATE' in df.columns:
        df['CLOSE_DATE_CLEAN'] = pd.to_datetime(df['CLOSE_DATE'], errors='coerce')
        df['CLOSE_DATE_CLEAN'] = df['CLOSE_DATE_CLEAN'].fillna(df['SUBMIT_DATE_CLEAN'])
    else:
        df['CLOSE_DATE_CLEAN'] = df['SUBMIT_DATE_CLEAN']

    df['duree_jours'] = (df['CLOSE_DATE_CLEAN'] - df['SUBMIT_DATE_CLEAN']).dt.days
    df['duree_jours'] = df['duree_jours'].fillna(0).apply(lambda x: x if x >= 0 else 0)

    def quick_cat(text):
        text = str(text).lower()
        if 'internet' in text or 'adsl' in text or '4g' in text or 'data' in text or 'flux' in text:
            return 'Problèmes Internet'
        elif 'appel' in text or 'numéro' in text or 'tonalité' in text or 'fixe' in text or 'ligne' in text:
            return 'Problèmes Téléphonie FIXE'
        else:
            return 'Autres Dérangements'
            
    df['TYPE_DERG_GROUP'] = df['TYPE_DERG'].apply(quick_cat)
    return df

df_processed = preprocess_clean_data(df_raw)

st.sidebar.markdown("###  Filtres de contrôle dynamique")

unique_provinces = ["Tous"] + list(df_processed['PROVINCE'].dropna().unique())
sel_province = st.sidebar.selectbox("1. Sélectionnez un état(PROVINCE):", unique_provinces)

unique_status = ["Tous"] + list(df_processed['TT_STATUS'].dropna().unique())
sel_status = st.sidebar.selectbox("2. État du billet (TT_STATUS):", unique_status)
unique_derg = ["Tous"] + list(df_processed['TYPE_DERG'].dropna().unique())
sel_derg = st.sidebar.selectbox("3. Le type exact du problème (TYPE_DERG):", unique_derg)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Plage horaire du rapport")
min_date = df_processed['SUBMIT_DATE_CLEAN'].min().date()
max_date = df_processed['SUBMIT_DATE_CLEAN'].max().date()

start_date = st.sidebar.date_input("Date de début:", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Date de fin :", max_date, min_value=min_date, max_value=max_date)

df_filtered = df_processed.copy()

if sel_province != "Tous":
    df_filtered = df_filtered[df_filtered['PROVINCE'] == sel_province]
if sel_status != "Tous":
    df_filtered = df_filtered[df_filtered['TT_STATUS'] == sel_status]
if sel_derg != "Tous":
    df_filtered = df_filtered[df_filtered['TYPE_DERG'] == sel_derg]

df_filtered = df_filtered[
    (df_filtered['SUBMIT_DATE_CLEAN'].dt.date >= start_date) & 
    (df_filtered['SUBMIT_DATE_CLEAN'].dt.date <= end_date)
]

st.markdown("### 📈Points de référence généraux du réseau")
kpi1, kpi2, kpi3 = st.columns(3)
total_records = len(df_filtered)

closed_count = len(df_filtered[df_filtered['TT_STATUS'].astype(str).str.lower().str.contains('close|clôturé|حل|terminé|1', na=False)])
res_rate = (closed_count / total_records * 100) if total_records > 0 else 100.0
avg_time = df_filtered['duree_jours'].mean() if total_records > 0 else 0.0

with kpi1:
    st.metric(label="📊Nombre total de plaintes durant cette période", value=f"{total_records:,}")
with kpi2:
    st.metric(label="✅ Taux de clôture et de résolution des tickets", value=f"{res_rate:.1f} %")
with kpi3:
    st.metric(label="⏳ Durée moyenne de réparation (jours)", value=f"{avg_time:.1f} jour")

st.markdown("---")

r1_c1, r1_c2 = st.columns(2)
with r1_c1:
    st.markdown("#### 🍩 1. Répartition relative des plaintes par État (PROVINCE)")
    if len(df_filtered) > 0:
        p_data = df_filtered['PROVINCE'].value_counts().reset_index()
        p_data.columns = ['Wilaya', 'Nombre']
        fig1 = px.pie(p_data.head(15), names='Wilaya', values='Nombre', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
    else:
        fig1 = px.pie(title="Pas de données")
    fig1.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

with r1_c2:
    st.markdown("#### 📊 2. Comparaison entre le statut du ticket et le type de panne (horizontale étendue)")
    fig2 = px.histogram(
        df_filtered, 
        y='TYPE_DERG_GROUP', 
        color='TT_STATUS', 
        barmode='group', 
        orientation='h',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig2.update_layout(height=350, yaxis_title=None, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
r2_c1, r2_c2 = st.columns(2)
with r2_c1:
    st.markdown("#### 📑 3. Les 10 provinces ayant enregistré le plus grand nombre de pannes de courant(DISTRICTINS)")
    if len(df_filtered) > 0:
        d_data = df_filtered['DISTRICTINS'].value_counts().reset_index().head(10)
        d_data.columns = ['District', 'Count']
        fig3 = px.bar(d_data, x='Count', y='District', orientation='h', color='Count', color_continuous_scale='Blues')
        fig3.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=10, r=10, t=10, b=10))
    else:
        fig3 = px.bar(title="Pas de données")
    st.plotly_chart(fig3, use_container_width=True)

with r2_c2:
    st.markdown("#### 📈 4.Chronologie systématique du flux de plaintes")
    if len(df_filtered) > 0:
        days_diff = (end_date - start_date).days
        rule = 'D' if days_diff <= 30 else 'W'
        
        ts_data = df_filtered.set_index('SUBMIT_DATE_CLEAN').resample(rule).size().reset_index()
        ts_data.columns = ['Temps', 'Nombre d\'appels']
        fig4 = px.line(ts_data, x='Temps', y='Nombre d\'appels', markers=True, color_discrete_sequence=['#0056b3'])
    else:
        fig4 = px.line(title="Pas de données")
    fig4.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
r3_c1, r3_c2 = st.columns(2)
with r3_c1:
    st.markdown("#### 📊 5. Volume réel des communications pour les 3 catégories approuvées")
    if len(df_filtered) > 0:
        v_data = df_filtered['TYPE_DERG_GROUP'].value_counts().reset_index()
        v_data.columns = ['Filière', 'Total']
        fig5 = px.bar(v_data, x='Filière', y='Total', color='Total', color_continuous_scale='Viridis')
    else:
        fig5 = px.bar(title="Pas de données")
    fig5.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)

with r3_c2:
    st.markdown("#### ⏱️ 6. Analyse de l'efficacité de la fermeture des tickets par jour(Box Plot)")
    if len(df_filtered) > 0:
        df_box = df_filtered[df_filtered['duree_jours'] <= 45]
        fig6 = px.box(df_box, x='TYPE_DERG_GROUP', y='duree_jours', color='TYPE_DERG_GROUP', color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        fig6 = px.box(title="Pas de données")
    fig6.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("<p style='text-align:center; color:gray; font-size:12px; margin-top:30px;'>Licence Data Science Graduation Project — Done by Haithem — Algérie Télécom © 2026</p>", unsafe_allow_html=True)