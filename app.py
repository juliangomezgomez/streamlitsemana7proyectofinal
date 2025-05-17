import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import io

st.set_page_config(layout="wide")
st.title("Detección de Anomalías por Cliente")

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

    for cliente, df_grupo in df_final.groupby('origen_hoja'):
        df_grupo_limpio = df_grupo[variables].replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_grupo_limpio) < 20:
            continue
        scaler = StandardScaler()
        datos_scaled = scaler.fit_transform(df_grupo_limpio)
        Q1 = df_grupo_limpio.quantile(0.25)
        Q3 = df_grupo_limpio.quantile(0.75)
        IQR = Q3 - Q1
        mask = ((df_grupo_limpio < (Q1 - 1.5 * IQR)) | (df_grupo_limpio > (Q3 + 1.5 * IQR)))
        contamination = mask.any(axis=1).mean()
        if (IQR == 0).all() or contamination == 0:
            contamination = 0.001
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(datos_scaled)
        pred = model.predict(datos_scaled)
        df_final.loc[df_grupo_limpio.index, 'anomalias'] = pred

    df_final['anomalias'] = df_final['anomalias'].fillna(1).astype(int)

    st.header("1. Selecciona el cliente a graficar")
    cliente_seleccionado = st.selectbox("Cliente", df_final['origen_hoja'].unique())
    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado]

    normales = df_cliente[df_cliente['anomalias'] == 1]
    anomalias = df_cliente[df_cliente['anomalias'] == -1]

    st.header("2. Gráficos de Presión, Volumen y Temperatura (toda la data disponible)")
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for i, var in enumerate(['Presion', 'Volumen', 'Temperatura']):
        axs[i].plot(normales['Fecha'], normales[var], label=f"{var} (Normal)")
        axs[i].scatter(anomalias['Fecha'], anomalias[var], label=f"{var} (Anómalo)", color='red', marker='x')

        y_max = normales[var].max()
        y_min = normales[var].min()
        axs[i].axhline(y=y_max, color='green', linestyle='--', alpha=0.4, label='Máximo normal')
        axs[i].axhline(y=y_min, color='orange', linestyle='--', alpha=0.4, label='Mínimo normal')

        fecha_media = normales['Fecha'].median()
        axs[i].text(fecha_media, y_max, f"{y_max:.2f}", va='bottom', ha='center', fontsize=8, color='green')
        axs[i].text(fecha_media, y_min, f"{y_min:.2f}", va='top', ha='center', fontsize=8, color='orange')

        axs[i].set_ylabel(var)
        axs[i].legend()
        axs[i].grid(True)
    axs[2].set_xlabel("Fecha")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Últimos valores disponibles")
    ultima_fecha = df_cliente['Fecha'].max()
    ultima_fila = df_cliente[df_cliente['Fecha'] == ultima_fecha].iloc[0]
    es_anomalo = ultima_fila['anomalias'] == -1
    color_valor = "red" if es_anomalo else "black"

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<span style='color:{color_valor}; font-size:24px'>Presión (psia): {ultima_fila['Presion']:.2f}</span>", unsafe_allow_html=True)
    col2.markdown(f"<span style='color:{color_valor}; font-size:24px'>Volumen (scf): {ultima_fila['Volumen']:.2f}</span>", unsafe_allow_html=True)
    col3.markdown(f"<span style='color:{color_valor}; font-size:24px'>Temperatura (°C): {ultima_fila['Temperatura']:.2f}</span>", unsafe_allow_html=True)

    st.header("3. Selecciona el rango de tiempo (horas)")
    rango_horas = st.slider("Rango de tiempo para analizar anomalías recientes", 0, 24, 2)

    st.header(f"4. Clientes con anomalías en las últimas {rango_horas} horas")
    clientes_recientes = []
    resumen_clientes = []
    for cliente, df_c in df_final.groupby('origen_hoja'):
        tmax = df_c['Fecha'].max()
        tmin = tmax - timedelta(hours=rango_horas)
        df_rango = df_c[df_c['Fecha'] >= tmin]
        total_anomalias = df_rango['anomalias'].eq(-1).sum()
        if total_anomalias > 0:
            clientes_recientes.append(cliente)
            resumen_clientes.append({"Cliente": cliente, "Anomalías recientes": total_anomalias})

    if clientes_recientes:
        st.write("Listado de clientes con anomalías recientes:")
        st.dataframe(pd.DataFrame(resumen_clientes))

        seleccionados = st.multiselect("Selecciona uno o más clientes para ver gráficas de las últimas 24 horas", clientes_recientes)

        for cliente_nuevo in seleccionados:
            df_cliente2 = df_final[df_final['origen_hoja'] == cliente_nuevo]
            fmax = df_cliente2['Fecha'].max()
            fmin = fmax - timedelta(hours=24)
            df_ultimas_24h_2 = df_cliente2[df_cliente2['Fecha'] >= fmin]
            normales2 = df_ultimas_24h_2[df_ultimas_24h_2['anomalias'] == 1]
            anomalias2 = df_ultimas_24h_2[df_ultimas_24h_2['anomalias'] == -1]

            st.subheader(f"Gráficas para {cliente_nuevo} (últimas 24 horas)")
            fig2, axs2 = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
            for i, var in enumerate(['Presion', 'Volumen', 'Temperatura']):
                axs2[i].plot(normales2['Fecha'], normales2[var], label=f"{var} (Normal)")
                axs2[i].scatter(anomalias2['Fecha'], anomalias2[var], label=f"{var} (Anómalo)", color='red', marker='x')
                axs2[i].set_ylabel(var)
                axs2[i].legend()
                axs2[i].grid(True)
            axs2[2].set_xlabel("Fecha")
            fig2.tight_layout()
            st.pyplot(fig2)

                                    st.subheader("5. Datos y descarga (CSV)")
            df_excel = df_ultimas_24h_2[['Fecha', 'Presion', 'Volumen', 'Temperatura']].copy()
            df_excel['Estado'] = df_ultimas_24h_2['anomalias'].map({1: 'Normal', -1: 'Anómalo'})
            st.dataframe(df_excel)
            

            st.download_button("Descargar CSV", data=df_excel.to_csv(index=False).encode('utf-8'),
                   file_name=f"{cliente_nuevo}_ultimas24h.csv",
                   mime="text/csv")
    else:
        st.info(f"No hay clientes con anomalías en las últimas {rango_horas} horas.")
