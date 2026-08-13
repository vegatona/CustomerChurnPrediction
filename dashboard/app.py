import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "churn_model.pkl"
)


# ============================================================
# CONSTANTES DEL MODELO
# ============================================================

CHURN_THRESHOLD = 0.30


# ============================================================
# CARGAR DATOS
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    return df


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    return model


# ============================================================
# PREPARACIÓN
# ============================================================

df = load_data()
model = load_model()


# ============================================================
# TÍTULO
# ============================================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    ### Sistema de análisis y predicción de abandono de clientes

    Este dashboard permite analizar el comportamiento de los clientes,
    identificar patrones relacionados con el abandono y estimar la
    probabilidad de que un cliente abandone el servicio.
    """
)

# SIDEBAR
st.sidebar.title("⚙️ Navegación")

page = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Resumen ejecutivo",
        "🔎 Análisis de abandono",
        "🔮 Predicción de Churn",
        "📈 Rendimiento del modelo",
        "💡 Insights y decisiones"
    ]
)

# DATOS GENERALES
total_customers = len(df)

churn_customers = (
    df["Churn"] == "Yes"
).sum()

active_customers = (
    df["Churn"] == "No"
).sum()

churn_rate = (
    churn_customers / total_customers
) * 100

# RESUMEN EJECUTIVO
if page == "🏠 Resumen ejecutivo":

    st.header("🏠 Resumen ejecutivo")

    st.markdown(
        """
        Esta sección presenta una visión general de la cartera
        de clientes y de la magnitud del problema de abandono.
        """
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total de clientes",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "⚠️ Clientes Churn",
            f"{churn_customers:,}"
        )

    with col3:

        st.metric(
            "✅ Clientes activos",
            f"{active_customers:,}"
        )

    with col4:

        st.metric(
            "📉 Tasa de abandono",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    # GRÁFICA GENERAL
    col1, col2 = st.columns(2)

    with col1:

        churn_data = pd.DataFrame(
            {
                "Estado": [
                    "Clientes activos",
                    "Clientes Churn"
                ],
                "Clientes": [
                    active_customers,
                    churn_customers
                ]
            }
        )

        fig = px.pie(
            churn_data,
            names="Estado",
            values="Clientes",
            title="Distribución de clientes"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        churn_summary = (
            df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_summary.columns = [
            "Churn",
            "Clientes"
        ]

        churn_summary["Estado"] = churn_summary[
            "Churn"
        ].map(
            {
                "Yes": "Churn",
                "No": "Activo"
            }
        )

        fig = px.bar(
            churn_summary,
            x="Estado",
            y="Clientes",
            title="Clientes por estado",
            text="Clientes"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # INTERPRETACIÓN
    st.subheader("🧠 Interpretación")

    st.info(
        f"""
        De los {total_customers:,} clientes analizados,
        {churn_customers:,} abandonaron el servicio.

        Esto representa una tasa de abandono de
        **{churn_rate:.2f}%**.

        Aproximadamente **1 de cada {round(100 / churn_rate)} clientes**
        presenta abandono en el conjunto de datos.
        """
    )

    st.subheader("🎯 Implicación para el negocio")

    st.success(
        """
        La empresa puede utilizar el sistema para identificar
        anticipadamente a los clientes con mayor probabilidad
        de abandono y dirigir las estrategias de retención
        hacia esos clientes.
        """
    )

# ANÁLISIS DE ABANDONO
elif page == "🔎 Análisis de abandono":

    st.header("🔎 Análisis de abandono")

    st.markdown(
        """
        Utiliza los filtros para explorar cómo cambia el comportamiento
        de abandono entre diferentes segmentos de clientes.
        """
    )

    # FILTROS
    st.sidebar.subheader("🎛️ Filtros")

    contracts = st.sidebar.multiselect(
        "Contrato",
        options=sorted(
            df["Contract"].unique()
        ),
        default=sorted(
            df["Contract"].unique()
        )
    )

    internet = st.sidebar.multiselect(
        "Servicio de Internet",
        options=sorted(
            df["InternetService"].unique()
        ),
        default=sorted(
            df["InternetService"].unique()
        )
    )

    payment = st.sidebar.multiselect(
        "Método de pago",
        options=sorted(
            df["PaymentMethod"].unique()
        ),
        default=sorted(
            df["PaymentMethod"].unique()
        )
    )

    # FILTRAR
    filtered_df = df[
        df["Contract"].isin(contracts)
        &
        df["InternetService"].isin(internet)
        &
        df["PaymentMethod"].isin(payment)
    ]

    st.write(
        f"Clientes incluidos en el análisis: "
        f"**{len(filtered_df):,}**"
    )

    # TASA DE CHURN
    st.subheader("📉 Tasa de abandono por contrato")

    contract_analysis = (
        filtered_df
        .groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index()
    )

    contract_analysis.columns = [
        "Contract",
        "ChurnRate"
    ]

    contract_analysis = contract_analysis.sort_values(
        "ChurnRate",
        ascending=False
    )

    fig = px.bar(
        contract_analysis,
        x="Contract",
        y="ChurnRate",
        text=contract_analysis["ChurnRate"].round(2),
        title="Porcentaje de abandono por tipo de contrato",
        labels={
            "Contract": "Contrato",
            "ChurnRate": "Tasa de Churn (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # CHURN POR INTERNET
    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌐 Churn por servicio de Internet"
        )

        internet_analysis = (
            filtered_df
            .groupby("InternetService")["Churn"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index()
        )

        internet_analysis.columns = [
            "InternetService",
            "ChurnRate"
        ]

        fig = px.bar(
            internet_analysis,
            x="InternetService",
            y="ChurnRate",
            text=internet_analysis[
                "ChurnRate"
            ].round(2),
            labels={
                "InternetService": "Servicio",
                "ChurnRate": "Churn (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # MÉTODO DE PAGO
    with col2:

        st.subheader(
            "💳 Churn por método de pago"
        )

        payment_analysis = (
            filtered_df
            .groupby("PaymentMethod")["Churn"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index()
        )

        payment_analysis.columns = [
            "PaymentMethod",
            "ChurnRate"
        ]

        payment_analysis = payment_analysis.sort_values(
            "ChurnRate",
            ascending=False
        )

        fig = px.bar(
            payment_analysis,
            x="ChurnRate",
            y="PaymentMethod",
            orientation="h",
            text=payment_analysis[
                "ChurnRate"
            ].round(2),
            labels={
                "PaymentMethod": "Método",
                "ChurnRate": "Churn (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # TENURE
    st.subheader(
        "📅 Relación entre antigüedad y abandono"
    )

    tenure_churn = (
        filtered_df
        .groupby("tenure")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index()
    )

    tenure_churn.columns = [
        "Tenure",
        "ChurnRate"
    ]

    fig = px.line(
        tenure_churn,
        x="Tenure",
        y="ChurnRate",
        markers=True,
        labels={
            "Tenure": "Meses de antigüedad",
            "ChurnRate": "Tasa de Churn (%)"
        },
        title="Tasa de abandono según antigüedad"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # CARGOS MENSUALES
    st.subheader(
        "💰 Cargos mensuales y abandono"
    )

    fig = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        labels={
            "Churn": "Estado",
            "MonthlyCharges": "Cargo mensual"
        },
        title="Distribución de cargos mensuales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# PREDICCIÓN
elif page == "🔮 Predicción de Churn":

    st.header("🔮 Predicción individual de abandono")

    st.markdown(
        """
        Introduce las características de un cliente para estimar
        su probabilidad de abandono.
        """
    )

    # DATOS DEL CLIENTE
    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Género",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Adulto mayor",
            [0, 1]
        )

        partner = st.selectbox(
            "Pareja",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependientes",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Antigüedad (meses)",
            min_value=0,
            max_value=72,
            value=2
        )

    with col2:

        phone = st.selectbox(
            "Servicio telefónico",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Múltiples líneas",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Seguridad en línea",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Respaldo en línea",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Protección del dispositivo",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Soporte técnico",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        contract = st.selectbox(
            "Contrato",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        paperless = st.selectbox(
            "Facturación electrónica",
            ["Yes", "No"]
        )

    with col3:

        payment_method = st.selectbox(
            "Método de pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col4:

        monthly = st.number_input(
            "Cargo mensual",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    total = st.number_input(
        "Cargos totales",
        min_value=0.0,
        value=140.0,
        step=10.0
    )

    st.divider()

    # PREDICCIÓN
    if st.button(
        "🔮 PREDECIR ABANDONO",
        use_container_width=True
    ):

        customer = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [senior],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly],
                "TotalCharges": [total]
            }
        )

        probability = model.predict_proba(
            customer
        )[0][1]

        prediction = (
            "Yes"
            if probability >= CHURN_THRESHOLD
            else "No"
        )

        percentage = probability * 100

        st.divider()

        st.subheader(
            "📊 Resultado de la evaluación"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Probabilidad de abandono",
                f"{percentage:.2f}%"
            )

        with col2:

            st.metric(
                "Umbral utilizado",
                f"{CHURN_THRESHOLD * 100:.0f}%"
            )

        with col3:

            if prediction == "Yes":

                st.metric(
                    "Predicción",
                    "⚠️ CHURN"
                )

            else:

                st.metric(
                    "Predicción",
                    "✅ NO CHURN"
                )

        # BARRA DE PROBABILIDAD
        st.progress(
            min(probability, 1.0)
        )

        # NIVEL DE RIESGO
        if probability >= 0.70:

            risk = "🔴 RIESGO MUY ALTO"

            message = (
                "El cliente presenta una probabilidad "
                "elevada de abandono."
            )

        elif probability >= 0.50:

            risk = "🟠 RIESGO ALTO"

            message = (
                "El cliente presenta señales importantes "
                "de posible abandono."
            )

        elif probability >= CHURN_THRESHOLD:

            risk = "🟡 RIESGO MODERADO"

            message = (
                "El cliente supera el umbral definido "
                "para activar una estrategia preventiva."
            )

        else:

            risk = "🟢 RIESGO BAJO"

            message = (
                "El cliente no supera el umbral definido "
                "para considerar abandono."
            )

        st.subheader(risk)

        st.info(message)

        # RECOMENDACIÓN
        if prediction == "Yes":

            st.warning(
                """
                ### 🎯 Recomendación de retención

                Se recomienda priorizar este cliente para
                una estrategia de retención.

                Algunas acciones posibles son:

                - Contacto personalizado.
                - Incentivos de permanencia.
                - Revisión del plan contratado.
                - Promociones personalizadas.
                - Seguimiento posterior.
                """
            )

        else:

            st.success(
                """
                ### ✅ Recomendación

                El cliente no supera el umbral de riesgo.
                Puede mantenerse dentro del seguimiento normal.
                """
            )

# RENDIMIENTO DEL MODELO
elif page == "📈 Rendimiento del modelo":

    st.header("📈 Rendimiento del modelo")

    st.markdown(
        """
        Comparación de los modelos supervisados evaluados durante
        el desarrollo del proyecto.
        """
    )

    # MODELOS
    model_results = pd.DataFrame(
        {
            "Modelo": [
                "Regresión Logística",
                "Árbol de Decisión",
                "Random Forest"
            ],
            "Accuracy": [
                0.8055,
                0.7984,
                0.8070
            ],
            "Precision": [
                0.6572,
                0.6347,
                0.6821
            ],
            "Recall": [
                0.5588,
                0.5668,
                0.5107
            ],
            "F1": [
                0.6040,
                0.5989,
                0.5841
            ]
        }
    )

    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    # COMPARACIÓN
    metrics_long = model_results.melt(
        id_vars="Modelo",
        var_name="Métrica",
        value_name="Valor"
    )

    fig = px.bar(
        metrics_long,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Comparación de modelos"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # OPTIMIZACIÓN
    st.divider()

    st.subheader(
        "🎯 Optimización del umbral"
    )

    threshold_results = pd.DataFrame(
        {
            "Umbral": [
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60
            ],
            "Accuracy": [
                0.7495,
                0.7644,
                0.7771,
                0.7899,
                0.8055,
                0.7991,
                0.7991
            ],
            "Precision": [
                0.5193,
                0.5432,
                0.5682,
                0.6021,
                0.6572,
                0.6784,
                0.7177
            ],
            "Recall": [
                0.7540,
                0.7059,
                0.6684,
                0.6150,
                0.5588,
                0.4626,
                0.4011
            ],
            "F1": [
                0.6150,
                0.6140,
                0.6143,
                0.6085,
                0.6040,
                0.5501,
                0.5146
            ]
        }
    )

    fig = px.line(
        threshold_results,
        x="Umbral",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ],
        markers=True,
        title="Comportamiento de las métricas según el umbral"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        """
        **Umbral seleccionado: 30%**

        Se seleccionó este umbral debido a que produce el mejor
        F1-Score dentro de los umbrales evaluados y aumenta
        considerablemente el Recall.

        Esto permite identificar una mayor cantidad de clientes
        que realmente presentan abandono.
        """
    )

    # MÉTRICAS FINALES
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "74.95%"
        )

    with col2:

        st.metric(
            "Precision",
            "51.93%"
        )

    with col3:

        st.metric(
            "Recall",
            "75.40%"
        )

    with col4:

        st.metric(
            "F1-Score",
            "61.50%"
        )

    # MATRIZ DE CONFUSIÓN
    st.subheader(
        "📊 Matriz de confusión — umbral 30%"
    )

    matrix = np.array(
        [
            [774, 261],
            [92, 282]
        ]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=[
            "Predicción: No Churn",
            "Predicción: Churn"
        ],
        y=[
            "Real: No Churn",
            "Real: Churn"
        ],
        title="Matriz de confusión"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        El modelo identificó correctamente 282 clientes que
        realmente abandonaron el servicio.

        Se produjeron 92 falsos negativos, es decir, clientes
        que abandonaron pero fueron clasificados como No Churn.

        Por esta razón se priorizó un umbral más bajo, buscando
        aumentar la capacidad del sistema para detectar clientes
        potencialmente propensos al abandono.
        """
    )

# INSIGHTS Y DECISIONES
elif page == "💡 Insights y decisiones":

    st.header("💡 Insights y decisiones de negocio")

    st.markdown(
        """
        El objetivo de esta sección es transformar los resultados
        del análisis y del modelo predictivo en acciones que puedan
        apoyar la toma de decisiones.
        """
    )

    # INSIGHT 1
    st.subheader("1️⃣ Nivel general de abandono")

    st.metric(
        "Tasa de Churn",
        f"{churn_rate:.2f}%"
    )

    st.write(
        f"""
        De los {total_customers:,} clientes analizados,
        {churn_customers:,} abandonaron el servicio.
        """
    )

    st.success(
        """
        **Decisión:** implementar un proceso de identificación
        temprana de clientes en riesgo para dirigir estrategias
        de retención antes de que ocurra el abandono.
        """
    )

    # INSIGHT 2
    st.subheader(
        "2️⃣ Contratos y permanencia"
    )

    contract_rates = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .sort_values(
            ascending=False
        )
    )

    highest_contract = contract_rates.index[0]
    highest_rate = contract_rates.iloc[0]

    st.write(
        f"""
        El segmento con mayor tasa de abandono dentro del
        dataset es **{highest_contract}**, con aproximadamente
        **{highest_rate:.2f}%** de Churn.
        """
    )

    st.success(
        """
        **Decisión:** priorizar estrategias de permanencia,
        incentivos y ofertas personalizadas para los segmentos
        contractuales con mayor riesgo.
        """
    )

    # INSIGHT 3
    st.subheader(
        "3️⃣ Servicio de Internet"
    )

    internet_rates = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .sort_values(
            ascending=False
        )
    )

    highest_internet = internet_rates.index[0]
    highest_internet_rate = internet_rates.iloc[0]

    st.write(
        f"""
        El servicio de Internet con mayor tasa de abandono
        dentro del dataset es **{highest_internet}**, con
        aproximadamente **{highest_internet_rate:.2f}%**.
        """
    )

    st.success(
        """
        **Decisión:** analizar la experiencia, precio y calidad
        del servicio en los segmentos con mayor abandono.
        """
    )

    # INSIGHT 4
    st.subheader(
        "4️⃣ Antigüedad del cliente"
    )

    low_tenure = df[
        df["tenure"] <= 12
    ]

    low_tenure_rate = (
        low_tenure["Churn"] == "Yes"
    ).mean() * 100

    st.write(
        f"""
        Entre los clientes con una antigüedad de hasta 12 meses,
        la tasa de abandono observada es de aproximadamente
        **{low_tenure_rate:.2f}%**.
        """
    )

    st.success(
        """
        **Decisión:** implementar seguimiento durante los primeros
        meses de la relación con el cliente y ofrecer incentivos
        de permanencia cuando exista riesgo elevado.
        """
    )

    # INSIGHT 5
    st.subheader(
        "5️⃣ Sistema predictivo"
    )

    st.write(
        """
        El modelo permite estimar la probabilidad individual
        de abandono y clasificar al cliente utilizando un umbral
        de decisión del 30%.
        """
    )

    st.success(
        """
        **Decisión:** utilizar el modelo como mecanismo de
        priorización. Los clientes que superen el umbral pueden
        ser enviados a un proceso de retención para contacto
        personalizado.
        """
    )

    # CONCLUSIÓN
    st.divider()

    st.header(
        "🎯 Conclusión ejecutiva"
    )

    st.info(
        """
        El sistema integra análisis de datos y aprendizaje
        supervisado para transformar información histórica de
        clientes en información útil para la toma de decisiones.

        El dashboard permite pasar de una visión descriptiva
        del abandono a una estrategia preventiva, identificando
        clientes con mayor probabilidad de Churn y permitiendo
        priorizar las acciones de retención.
        """
    )