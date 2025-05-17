import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(layout="wide", page_title="Detección de Anomalías por Cliente")

st.title("🔍 Detección de Anomalías en Datos por Cliente")

# Subir archivo
uploaded_file = st.file_uploader("📤 Carga tu archivo Excel", type=["xlsx"])

if uploaded_file:
    hojas = pd.read_excel(uploaded_file, sheet_name=None)
    dataframes = []

    for nombre_hoja, df in hojas.items():
        if not df.empty:
            df['origen_hoja'] = nombre_hoja
            dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)

    # Asegurar que 'Fecha' sea datetime
    df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])
    df_final = df_final.sort_values(by='Fecha')

    # Variables
    variables = ['Presion', 'Temperatura', 'Volumen']
    df_final['anomalias'] = np.nan

    # Detección de anomalías
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

    # Selección de cliente
    clientes_con_anomalias = df_final[df_final['anomalias'] == -1]['origen_hoja'].unique().tolist()
    cliente_seleccionado = st.selectbox("🧾 Selecciona un cliente", sorted(df_final['origen_hoja'].unique()))

    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado]

    # Último valor (reloj)
    ultimo = df_cliente.sort_values('Fecha').iloc[-1]
    st.subheader("⏱ Últimos valores disponibles")
    col1, col2, col3 = st.columns(3)
    col1.metric("Presión", f"{ultimo['Presion']}")
    col2.metric("Temperatura", f"{ultimo['Temperatura']}")
    col3.metric("Volumen", f"{ultimo['Volumen']}")

    st.subheader("📊 Gráficas de Variables con Líneas de Máximos y Mínimos (sin anomalías)")

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    df_normal = df_cliente[df_cliente['anomalias'] == 1]

    for i, variable in enumerate(variables):
        ax = axs[i]
        sns.scatterplot(data=df_cliente, x='Fecha', y=variable,
                        hue='anomalias', palette={1: 'blue', -1: 'red'}, ax=ax)
        ax.axhline(df_normal[variable].max(), color='green', linestyle='--', label='Máximo (sin anomalías)')
        ax.axhline(df_normal[variable].min(), color='orange', linestyle='--', label='Mínimo (sin anomalías)')
        ax.set_title(f'{variable} - {cliente_seleccionado}')
        ax.legend()

    st.pyplot(fig)

    # Último día de datos
    max_fecha = df_cliente['Fecha'].max()
    fecha_inicio = max_fecha.normalize()

    df_ultimo_dia = df_cliente[df_cliente['Fecha'] >= fecha_inicio]
    df_normales_dia = df_ultimo_dia[df_ultimo_dia['anomalias'] == 1]
    df_anomalos_dia = df_ultimo_dia[df_ultimo_dia['anomalias'] == -1]

    st.subheader("📅 Datos del último día")
    with st.expander("📈 Gráficas de datos normales"):
        for var in variables:
            st.line_chart(df_normales_dia.set_index('Fecha')[var])

    with st.expander("🚨 Gráficas de datos anómalos"):
        for var in variables:
            st.line_chart(df_anomalos_dia.set_index('Fecha')[var])

    st.markdown("### 📥 Descargar datos del último día")
    col_a, col_b = st.columns(2)

    def generar_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    col_a.download_button("⬇️ Descargar normales", generar_csv(df_normales_dia),
                          file_name=f"{cliente_seleccionado}_normales.csv", mime='text/csv')
    col_b.download_button("⬇️ Descargar anómalos", generar_csv(df_anomalos_dia),
                          file_name=f"{cliente_seleccionado}_anomalos.csv", mime='text/csv')

    # Alerta de anomalías en últimas 2 horas de datos
    st.subheader("🚨 Alerta de anomalías en últimas 2 horas (según los datos)")

    ultimas_2h = df_cliente['Fecha'].max() - pd.Timedelta(hours=2)
    anomalias_2h = df_cliente[(df_cliente['Fecha'] >= ultimas_2h) & (df_cliente['anomalias'] == -1)]

    if not anomalias_2h.empty:
        st.error(f"⚠️ {cliente_seleccionado} tiene anomalías en las últimas 2 horas.")
        st.dataframe(anomalias_2h)
    else:
        st.success("✅ No se encontraron anomalías en las últimas 2 horas para este cliente.")

else:
    st.info("Por favor sube un archivo Excel con varias hojas, una por cliente.")
