import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(page_title="Detección de Anomalías", layout="wide")
st.title("🔍 Detección de Anomalías por Cliente")

# 1. Subir archivo Excel
archivo = st.file_uploader("📁 Sube un archivo Excel (.xlsx)", type=["xlsx"])

if archivo:
    hojas = pd.read_excel(archivo, sheet_name=None)
    dataframes = []

    for nombre_hoja, df in hojas.items():
        if not df.empty:
            df['origen_hoja'] = nombre_hoja
            dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)
    variables = ['Presion', 'Temperatura', 'Volumen']
    df_final['anomalias'] = np.nan

    # 2. Procesamiento por cliente
    for cliente, df_grupo in df_final.groupby('origen_hoja'):
        df_grupo_limpio = df_grupo[variables].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_grupo_limpio) < 20:
            continue

        scaler = StandardScaler()
        datos_scaled = scaler.fit_transform(df_grupo_limpio)

        Q1 = df_grupo_limpio.quantile(0.25)
        Q3 = df_grupo_limpio.quantile(0.75)
        IQR = Q3 - Q1
        outliers_mask = ((df_grupo_limpio < (Q1 - 1.5 * IQR)) | (df_grupo_limpio > (Q3 + 1.5 * IQR)))
        contamination = outliers_mask.any(axis=1).mean()

        if (IQR == 0).all() or contamination == 0:
            contamination = 0.001

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(datos_scaled)
        predicciones = model.predict(datos_scaled)

        df_final.loc[df_grupo_limpio.index, 'anomalias'] = predicciones

    df_final['anomalias'] = df_final['anomalias'].fillna(1).astype(int)

    st.success("✅ Detección completada")
    
    # 3. Filtros de selección
    cliente_sel = st.selectbox("Selecciona un cliente", df_final["origen_hoja"].unique())
    variable_sel = st.selectbox("Selecciona una variable", variables)

    df_cliente = df_final[df_final["origen_hoja"] == cliente_sel]

    # 4. Gráfico de anomalías
    st.subheader(f"📈 Gráfico de {variable_sel} - Cliente: {cliente_sel}")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_cliente,
        x="Fecha",
        y=variable_sel,
        hue="anomalias",
        palette={1: "blue", -1: "red"},
        ax=ax,
        s=50
    )
    ax.set_title(f"Anomalías en {variable_sel}")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # 5. Descarga del resultado
    st.subheader("📥 Descargar resultados")
    buffer = BytesIO()
    df_final.to_excel(buffer, index=False)
    st.download_button(
        label="Descargar Excel con anomalías",
        data=buffer,
        file_name="resultado_anomalias.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
