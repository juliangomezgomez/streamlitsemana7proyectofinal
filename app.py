import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

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
    for var in ['Presion', 'Volumen', 'Temperatura']:
        fig = go.Figure()
        if not normales.empty:
            fig.add_trace(go.Scatter(x=normales['Fecha'], y=normales[var], mode='lines', name=f'{var} (Normal)'))
        if not anomalias.empty:
            fig.add_trace(go.Scatter(x=anomalias['Fecha'], y=anomalias[var], mode='markers', name=f'{var} (Anómalo)', marker=dict(color='red', symbol='x')))

        if not normales.empty:
            y_max = normales[var].max()
            y_min = normales[var].min()
            fig.add_hline(y=y_max, line_dash='dash', line_color='green', annotation_text='Máximo normal', annotation_position='top left')
            fig.add_hline(y=y_min, line_dash='dash', line_color='orange', annotation_text='Mínimo normal', annotation_position='bottom left')

        fig.update_layout(title=f'{var} - Cliente: {cliente_seleccionado}', xaxis_title='Fecha', yaxis_title=var, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

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
            for var in ['Presion', 'Volumen', 'Temperatura']:
                fig = go.Figure()
                if not normales2.empty:
                    fig.add_trace(go.Scatter(x=normales2['Fecha'], y=normales2[var], mode='lines', name=f'{var} (Normal)'))
                if not anomalias2.empty:
                    fig.add_trace(go.Scatter(x=anomalias2['Fecha'], y=anomalias2[var], mode='markers', name=f'{var} (Anómalo)', marker=dict(color='red', symbol='x')))
                fig.update_layout(title=f'{var} - Cliente: {cliente_nuevo}', xaxis_title='Fecha', yaxis_title=var, template='plotly_white', hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"No hay clientes con anomalías en las últimas {rango_horas} horas.")

    st.divider()
    st.header("6. Consumo mensual por cliente y desviaciones")

    st.divider()
    st.header("7. Exportar variable de un cliente a CSV")
    cliente_exportar = st.selectbox("Selecciona un cliente para exportar", df_final['origen_hoja'].unique())
    variable_exportar = st.selectbox("Selecciona una variable", variables)

    fecha_min_exp, fecha_max_exp = st.date_input("Selecciona rango de fechas", value=(df_final['Fecha'].min(), df_final['Fecha'].max()))
    df_exp = df_final[(df_final['origen_hoja'] == cliente_exportar) &
                      (df_final['Fecha'].dt.date >= fecha_min_exp) &
                      (df_final['Fecha'].dt.date <= fecha_max_exp)][['Fecha', variable_exportar, 'anomalias']].copy()
    df_exp['Cliente'] = cliente_exportar
    df_exp['Variable'] = variable_exportar
    df_exp['Estado'] = df_exp['anomalias'].map({1: 'Normal', -1: 'Anómalo'})
    df_exp = df_exp[['Fecha', 'Cliente', 'Variable', variable_exportar, 'Estado']]
    st.dataframe(df_exp)
    csv_data = df_exp.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv_data,
        file_name=f"{cliente_exportar}_{variable_exportar}.csv",
        mime="text/csv"
    )
    ultimo_mes = df_final['Fecha'].max() - pd.DateOffset(days=30)
    df_mes = df_final[df_final['Fecha'] >= ultimo_mes]
    resumen_consumo = []
    for cliente, df_c in df_mes.groupby('origen_hoja'):
        dias = df_c['Fecha'].dt.normalize().nunique()
        vol_mensual = df_c['Volumen'].mean() * dias if dias > 0 else 0
        presion_mensual = df_c['Presion'].mean()
        total_anomalias = df_c['anomalias'].eq(-1).sum()

        df_cliente_todo = df_final[df_final['origen_hoja'] == cliente]
        ultima_fecha = df_cliente_todo['Fecha'].max()
        actual = df_cliente_todo[df_cliente_todo['Fecha'] == ultima_fecha].iloc[-1]

        vol_actual = actual['Volumen']
        presion_actual = actual['Presion']

        vol_estado = 'Normal'
        if vol_mensual > 0:
            if vol_actual > 1.8 * vol_mensual:
                vol_estado = '80% más alto'
            elif vol_actual < 0.2 * vol_mensual:
                vol_estado = '80% más bajo'

        presion_estado = 'Normal'
        if presion_mensual > 0:
            if presion_actual > 1.8 * presion_mensual:
                presion_estado = '80% más alto'
            elif presion_actual < 0.2 * presion_mensual:
                presion_estado = '80% más bajo'

        resumen_consumo.append({
            'Cliente': cliente,
            'Consumo estimado (scf)': round(vol_mensual, 2),
            'Anomalías en el mes': total_anomalias,
            'Estado Volumen': vol_estado,
            'Estado Presion': presion_estado
        })

    df_resumen_consumo = pd.DataFrame(resumen_consumo).sort_values(by='Consumo estimado (scf)', ascending=False)
    st.dataframe(df_resumen_consumo)
    csv_consumo = df_resumen_consumo.to_csv(index=False)
    st.download_button(
        label="Descargar tabla de consumo mensual",
        data=csv_consumo,
        file_name="consumo_mensual_clientes.csv",
        mime="text/csv"
    )

    st.divider()
    st.header("8. Filtrar clientes con desviaciones extremas")
    df_desv = df_resumen_consumo[(df_resumen_consumo['Estado Volumen'] != 'Normal') | (df_resumen_consumo['Estado Presion'] != 'Normal')]
    st.dataframe(df_desv)
    csv_desv = df_desv.to_csv(index=False)
    st.download_button(
        label="Descargar clientes con desviaciones",
        data=csv_desv,
        file_name="clientes_con_desviaciones.csv",
        mime="text/csv"
    )

    

    
