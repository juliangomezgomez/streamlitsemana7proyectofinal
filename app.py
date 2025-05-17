import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Análisis de Anomalías por Cliente")

uploaded_file = st.file_uploader("Sube tu archivo Excel con datos", type=["xlsx"])

if uploaded_file is not None:
    hojas = pd.read_excel(uploaded_file, sheet_name=None)
    dataframes = []
    for nombre_hoja, df in hojas.items():
        if not df.empty:
            df['origen_hoja'] = nombre_hoja
            dataframes.append(df)
    df_final = pd.concat(dataframes, ignore_index=True)

    # Convertir 'Fecha' a datetime si no lo está
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
        outliers_mask = ((df_grupo_limpio < (Q1 - 1.5 * IQR)) | (df_grupo_limpio > (Q3 + 1.5 * IQR)))
        contamination = outliers_mask.any(axis=1).mean()

        if (IQR == 0).all() or contamination == 0:
            contamination = 0.001

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(datos_scaled)
        predicciones = model.predict(datos_scaled)

        df_final.loc[df_grupo_limpio.index, 'anomalias'] = predicciones

    df_final['anomalias'] = df_final['anomalias'].fillna(1).astype(int)

    clientes = df_final['origen_hoja'].unique()
    cliente_seleccionado = st.selectbox("Selecciona el cliente", clientes)

    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado].sort_values('Fecha')

    # Parámetro para definir cuántos datos mostrar
    n = st.number_input("Número de datos anómalos recientes a mostrar", min_value=1, max_value=500, value=50)

    # Filtrar anomalías
    df_anomalias = df_cliente[df_cliente['anomalias'] == -1]

    # Últimos n datos anómalos (ordenados por fecha descendente)
    ultimos_anomalos = df_anomalias.sort_values('Fecha', ascending=False).head(n)

    st.subheader(f"Últimos {n} datos anómalos para {cliente_seleccionado}")
    st.dataframe(ultimos_anomalos)

    # Revisar si hay anomalías en las últimas 2 horas
    ahora = pd.Timestamp.now()
    dos_horas_atras = ahora - pd.Timedelta(hours=2)
    anomalias_ultimas_2h = df_anomalias[(df_anomalias['Fecha'] >= dos_horas_atras) & (df_anomalias['Fecha'] <= ahora)]

    if not anomalias_ultimas_2h.empty:
        st.error(f"⚠️ ALERTA: Se encontraron {len(anomalias_ultimas_2h)} anomalías en las últimas 2 horas!")
        st.dataframe(anomalias_ultimas_2h)
    else:
        st.success("No se detectaron anomalías en las últimas 2 horas.")

    # Mostrar gráficos con líneas de máximo, mínimo y anomalías
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    colores = {1: 'blue', -1: 'red'}

    for i, variable in enumerate(variables):
        sns.scatterplot(
            data=df_cliente,
            x='Fecha',
            y=variable,
            hue='anomalias',
            palette=colores,
            ax=axs[i],
            legend=(i == 0),
            s=40
        )
        max_val = df_cliente[variable].max()
        min_val = df_cliente[variable].min()
        axs[i].axhline(max_val, color='green', linestyle='--', label='Máximo')
        axs[i].axhline(min_val, color='orange', linestyle='--', label='Mínimo')

        axs[i].set_title(f'{variable} vs Fecha')
        axs[i].set_xlabel('Fecha')
        axs[i].set_ylabel(variable)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].grid(True)

        if i == 0:
            axs[i].legend()

    plt.tight_layout()
    st.pyplot(fig)
