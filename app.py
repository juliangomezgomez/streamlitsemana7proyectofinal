import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Detección de Anomalías por Cliente")

uploaded_file = st.file_uploader("Carga tu archivo Excel con datos por cliente", type=["xlsx"])

if uploaded_file is not None:
    hojas = pd.read_excel(uploaded_file, sheet_name=None)
    dataframes = []
    for nombre_hoja, df in hojas.items():
        if not df.empty:
            df['origen_hoja'] = nombre_hoja
            dataframes.append(df)
    df_final = pd.concat(dataframes, ignore_index=True)
    
    # Asegurarse que 'Fecha' sea datetime
    df_final['Fecha'] = pd.to_datetime(df_final['Fecha'], errors='coerce')
    df_final = df_final.dropna(subset=['Fecha'])

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

    # Selector de cliente
    clientes = df_final['origen_hoja'].unique()
    cliente_seleccionado = st.selectbox("Selecciona un cliente", clientes)

    df_cliente = df_final[df_final['origen_hoja'] == cliente_seleccionado]

    # Gráficas con líneas max y min y anomalías
    fig, axs = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)

    for i, var in enumerate(variables):
        ax = axs[i] if len(variables) > 1 else axs
        sns.scatterplot(
            data=df_cliente,
            x='Fecha',
            y=var,
            hue='anomalias',
            palette={1: 'blue', -1: 'red'},
            ax=ax,
            legend=(i == 0),
            s=40
        )
        ax.set_title(f'{var} vs Fecha - Cliente: {cliente_seleccionado}')
        ax.set_ylabel(var)
        ax.grid(True)

        # Líneas de máximo y mínimo
        ax.axhline(df_cliente[var].max(), color='green', linestyle='--', label='Máximo' if i == 0 else "")
        ax.axhline(df_cliente[var].min(), color='orange', linestyle='--', label='Mínimo' if i == 0 else "")
        if i == 0:
            ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Anomalías en últimas 2 horas de datos
    fecha_max_global = df_final['Fecha'].max()
    dos_horas_atras = fecha_max_global - pd.Timedelta(hours=2)

    anomalias_ultimas_2h = df_final[
        (df_final['anomalias'] == -1) &
        (df_final['Fecha'] >= dos_horas_atras) &
        (df_final['Fecha'] <= fecha_max_global)
    ]

    clientes_con_anomalias_ultimas_2h = anomalias_ultimas_2h.groupby('origen_hoja').size().reset_index(name='Cantidad Anomalías')

    st.subheader("Clientes con anomalías en las últimas 2 horas de datos")

    if clientes_con_anomalias_ultimas_2h.empty:
        st.write("No se detectaron anomalías en las últimas 2 horas para ningún cliente.")
    else:
        st.dataframe(clientes_con_anomalias_ultimas_2h)

