import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import io

st.set_page_config(layout="wide")
st.title("🚀 Detección de Anomalías por Cliente")

# Cargar archivo
archivo = st.file_uploader("Sube un archivo Excel con hojas por cliente", type="xlsx")

if archivo:
    hojas = pd.read_excel(archivo, sheet_name=None)
    dataframes = []

    for nombre_hoja, df in hojas.items():
        if not df.empty:
            df['origen_hoja'] = nombre_hoja
            dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)
    df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])

    variables = ['Presion', 'Temperatura', 'Volumen']
    df_final['anomalias'] = np.nan

    # Detección de anomalías por cliente
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

    # Filtrado por cliente
    clientes_con_anomalias = df_final[df_final['anomalias'] == -1]['origen_hoja'].unique().tolist()
    clientes_ultimas_2h = []
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        cliente_seleccionado = st.selectbox("Selecciona un cliente:", df_final['origen_hoja'].unique())

    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado]

    # Obtener último día
    fecha_max = df_cliente['Fecha'].max()
    fecha_min = fecha_max - timedelta(days=1)
    df_ultimo_dia = df_cliente[df_cliente['Fecha'] >= fecha_min]

    # Datos normales y anómalos
    normales = df_ultimo_dia[df_ultimo_dia['anomalias'] == 1]
    anomalias = df_ultimo_dia[df_ultimo_dia['anomalias'] == -1]

    # Mostrar relojes
    with col2:
        st.metric("Presión actual", value=round(df_cliente['Presion'].iloc[-1], 2))
        st.metric("Volumen actual", value=round(df_cliente['Volumen'].iloc[-1], 2))
        st.metric("Temperatura actual", value=round(df_cliente['Temperatura'].iloc[-1], 2))

    # Gráfico combinado
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for var in variables:
            ax.plot(normales['Fecha'], normales[var], label=f"{var} (Normal)")
            ax.scatter(anomalias['Fecha'], anomalias[var], label=f"{var} (Anómalo)", color='red', marker='x')

            # Calcular y trazar líneas de máximo y mínimo sin anomalías
            y_max = normales[var].max()
            y_min = normales[var].min()
            ax.axhline(y=y_max, color='green', linestyle='--', alpha=0.3)
            ax.axhline(y=y_min, color='orange', linestyle='--', alpha=0.3)

        ax.set_title(f"Variables para {cliente_seleccionado} - Último día")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Desplegables con datos normales y anómalos
    with st.expander("Datos normales del último día"):
        st.dataframe(normales[['Fecha'] + variables])

    with st.expander("Datos anómalos del último día"):
        st.dataframe(anomalias[['Fecha'] + variables])

    # Datos anómalos últimas 2 horas de datos
    for cliente, df_cliente in df_final.groupby('origen_hoja'):
        if df_cliente['anomalias'].eq(-1).sum() == 0:
            continue
        max_fecha = df_cliente['Fecha'].max()
        min_fecha = max_fecha - timedelta(hours=2)
        df_reciente = df_cliente[(df_cliente['Fecha'] >= min_fecha)]
        if (df_reciente['anomalias'] == -1).any():
            clientes_ultimas_2h.append(cliente)

    st.sidebar.subheader("🚨 Clientes con anomalías recientes")
    if clientes_ultimas_2h:
        st.sidebar.write("<br>".join(clientes_ultimas_2h), unsafe_allow_html=True)
    else:
        st.sidebar.write("No hay clientes con anomalías en las últimas 2 horas de datos.")

    # Botón para descargar datos filtrados
    cliente_df_filtrado = df_final[df_final['origen_hoja'] == cliente_seleccionado]
    csv = cliente_df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📂 Descargar datos del cliente",
        data=csv,
        file_name=f"datos_{cliente_seleccionado}.csv",
        mime='text/csv')
