import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import io

st.set_page_config(layout="wide")
st.title("\ud83d\udcca Detecci\u00f3n de Anomal\u00edas por Cliente")

archivo = st.file_uploader("\ud83d\udcc1 Sube un archivo Excel con hojas por cliente", type="xlsx")

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

    st.header("1\ufe0f\ufe0f Selecciona el cliente a graficar")
    cliente_seleccionado = st.selectbox("Cliente", df_final['origen_hoja'].unique())
    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado]

    fecha_max = df_cliente['Fecha'].max()
    fecha_min = fecha_max - timedelta(hours=24)
    df_ultimas_24h = df_cliente[df_cliente['Fecha'] >= fecha_min]
    normales = df_ultimas_24h[df_ultimas_24h['anomalias'] == 1]
    anomalias = df_ultimas_24h[df_ultimas_24h['anomalias'] == -1]

    st.header("2\ufe0f\ufe0f Gr\u00e1ficos de Presi\u00f3n, Volumen y Temperatura")
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for i, var in enumerate(['Presion', 'Volumen', 'Temperatura']):
        axs[i].plot(normales['Fecha'], normales[var], label=f"{var} (Normal)")
        axs[i].scatter(anomalias['Fecha'], anomalias[var], label=f"{var} (An\u00f3malo)", color='red', marker='x')

        y_max = normales[var].max()
        y_min = normales[var].min()
        axs[i].axhline(y=y_max, color='green', linestyle='--', alpha=0.4, label='M\u00e1ximo normal')
        axs[i].axhline(y=y_min, color='orange', linestyle='--', alpha=0.4, label='M\u00ednimo normal')

        fecha_media = normales['Fecha'].median()
        axs[i].text(fecha_media, y_max, f"{y_max:.2f}", va='bottom', ha='center', fontsize=8, color='green')
        axs[i].text(fecha_media, y_min, f"{y_min:.2f}", va='top', ha='center', fontsize=8, color='orange')

        axs[i].set_ylabel(var)
        axs[i].legend()
        axs[i].grid(True)
    axs[2].set_xlabel("Fecha")
    fig.tight_layout()
    st.pyplot(fig)

    # Valores actuales
    st.markdown("#### \ud83d\udd52 \u00daltimos valores disponibles:")
    ultima_fecha = df_cliente['Fecha'].max()
    ultima_fila = df_cliente[df_cliente['Fecha'] == ultima_fecha].iloc[0]
    es_anomalo = ultima_fila['anomalias'] == -1
    color_valor = "red" if es_anomalo else "black"

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<span style='color:{color_valor}; font-size:24px'>Presi\u00f3n (psia): {ultima_fila['Presion']:.2f}</span>", unsafe_allow_html=True)
    col2.markdown(f"<span style='color:{color_valor}; font-size:24px'>Volumen (scf): {ultima_fila['Volumen']:.2f}</span>", unsafe_allow_html=True)
    col3.markdown(f"<span style='color:{color_valor}; font-size:24px'>Temperatura (\u00b0C): {ultima_fila['Temperatura']:.2f}</span>", unsafe_allow_html=True)

    st.header("3\ufe0f\ufe0f Selecciona el rango de tiempo (horas)")
    rango_horas = st.slider("Rango de tiempo para analizar anomal\u00edas recientes", 0, 24, 2)

    st.header("4\ufe0f\ufe0f Clientes con anomal\u00edas en las \u00faltimas {} horas".format(rango_horas))
    clientes_recientes = []
    for cliente, df_c in df_final.groupby('origen_hoja'):
        if df_c['anomalias'].eq(-1).sum() == 0:
            continue
        tmax = df_c['Fecha'].max()
        tmin = tmax - timedelta(hours=rango_horas)
        df_rango = df_c[df_c['Fecha'] >= tmin]
        if (df_rango['anomalias'] == -1).any():
            clientes_recientes.append(cliente)

    if clientes_recientes:
        cliente_nuevo = st.selectbox("Selecciona un cliente con anomal\u00edas recientes", clientes_recientes)
        df_cliente2 = df_final[df_final['origen_hoja'] == cliente_nuevo]
        fmax = df_cliente2['Fecha'].max()
        fmin = fmax - timedelta(hours=24)
        df_ultimas_24h_2 = df_cliente2[df_cliente2['Fecha'] >= fmin]
        normales2 = df_ultimas_24h_2[df_ultimas_24h_2['anomalias'] == 1]
        anomalias2 = df_ultimas_24h_2[df_ultimas_24h_2['anomalias'] == -1]

        st.subheader(f"\ud83d\udcc8 Gr\u00e1ficas para {cliente_nuevo}")
        fig2, axs2 = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        for i, var in enumerate(['Presion', 'Volumen', 'Temperatura']):
            axs2[i].plot(normales2['Fecha'], normales2[var], label=f"{var} (Normal)")
            axs2[i].scatter(anomalias2['Fecha'], anomalias2[var], label=f"{var} (An\u00f3malo)", color='red', marker='x')
            axs2[i].set_ylabel(var)
            axs2[i].legend()
            axs2[i].grid(True)
        axs2[2].set_xlabel("Fecha")
        fig2.tight_layout()
        st.pyplot(fig2)

        st.subheader("5\ufe0f\ufe0f Descarga de datos (Excel)")
        df_excel = pd.DataFrame({'Fecha': df_ultimas_24h_2['Fecha']})
        for var in variables:
            df_excel[f'{var}_Normal'] = normales2.set_index('Fecha')[var].reindex(df_excel['Fecha'])
            df_excel[f'{var}_Anomalo'] = anomalias2.set_index('Fecha')[var].reindex(df_excel['Fecha'])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_excel.to_excel(writer, index=False, sheet_name='Datos_Cliente')
        output.seek(0)

        st.download_button("\ud83d\udcc5 Descargar Excel", data=output,
                           file_name=f"{cliente_nuevo}_ultimas24h.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info(f"\u274c No hay clientes con anomal\u00edas en las \u00faltimas {rango_horas} horas.")
