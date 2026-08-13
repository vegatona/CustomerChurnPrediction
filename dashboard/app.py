import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

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

# Umbral utilizado para activar una alerta de Churn
CHURN_THRESHOLD = 0.30


# ==========================================================
# CONFIGURACIÓN DE STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# ESTILOS PERSONALIZADOS
# ==========================================================

st.markdown(
    """
    <style>

    /* ======================================================
       FONDO GENERAL
       ====================================================== */

    .stApp {
        background-color: #f5f7fb;
    }


    /* ======================================================
       TÍTULOS
       ====================================================== */

    h1 {
        font-weight: 800;
    }

    h2 {
        font-weight: 700;
    }

    h3 {
        font-weight: 650;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }


    /* ------------------------------------------------------
       TÍTULOS DEL SIDEBAR
       ------------------------------------------------------ */

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }


    /* ------------------------------------------------------
       TEXTO GENERAL DEL SIDEBAR
       ------------------------------------------------------ */

    section[data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
    }


    /* ======================================================
       NAVEGACIÓN
       ====================================================== */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] {
        gap: 5px;
        width: 100%;
    }


    /* Cada opción */

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label {
        background-color: transparent !important;
        border-radius: 10px;
        padding: 9px 10px;
        margin: 2px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
    }


    /* Hover */

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:hover {
        background-color: #1f2937 !important;
    }


    /* Texto */

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label p {
        color: #f1f5f9 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }


    /* Opción seleccionada */

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) {
        background-color: #4f46e5 !important;
        box-shadow: 0 3px 10px rgba(79, 70, 229, 0.30);
    }


    /* Texto seleccionado */

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }


    /* ======================================================
       TARJETAS
       ====================================================== */

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }


    .info-card h3 {
        margin-top: 0;
    }


    /* ======================================================
       DESCRIPCIONES
       ====================================================== */

    .description-card {
        background: #ffffff;
        border-left: 4px solid #4f46e5;
        padding: 16px 20px;
        margin: 10px 0 25px 0;
        border-radius: 8px;
        color: #374151;
        line-height: 1.5;
    }


    /* ======================================================
       TARJETA ÉXITO
       ====================================================== */

    .success-card {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        color: #065f46;
    }


    /* ======================================================
       TARJETA ADVERTENCIA
       ====================================================== */

    .warning-card {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        color: #92400e;
    }


    /* ======================================================
       TARJETA PELIGRO
       ====================================================== */

    .danger-card {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
        color: #991b1b;
    }


    /* ======================================================
       TARJETA MODELO
       ====================================================== */

    .model-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        line-height: 1.5;
    }


    /* ======================================================
       MÉTRICAS
       ====================================================== */

    .metric-title {
        font-size: 14px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }


    /* ======================================================
       SEPARADOR
       ====================================================== */

    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CARGAR DATASET
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


# ==========================================================
# CARGAR MODELO
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_FILE)


# ==========================================================
# CARGAR RECURSOS
# ==========================================================

try:

    df = load_data()
    model = load_model()

except Exception as e:

    st.error(
        "❌ No fue posible cargar el dataset o el modelo."
    )

    st.code(str(e))

    st.stop()


# ==========================================================
# IDENTIFICAR MODELO
# ==========================================================

def get_model_name(model):

    try:

        classifier = model.named_steps.get(
            "classifier"
        )

        if classifier is not None:

            model_class = classifier.__class__.__name__

            model_names = {
                "LogisticRegression":
                    "Regresión Logística",

                "DecisionTreeClassifier":
                    "Árbol de Decisión",

                "RandomForestClassifier":
                    "Random Forest"
            }

            return model_names.get(
                model_class,
                model_class
            )

    except Exception:

        pass

    return "Modelo supervisado"


model_name = get_model_name(model)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def show_description(title, text):

    st.markdown(
        f"""
        <div class="description-card">
            <strong>{title}</strong><br><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def risk_information(probability):

    if probability >= 0.70:

        return (
            "🔴 RIESGO MUY ALTO",
            "danger",
            "La probabilidad estimada de abandono es muy elevada. "
            "El cliente debe ser considerado prioritario dentro "
            "de las estrategias de retención."
        )

    elif probability >= 0.50:

        return (
            "🟠 RIESGO ALTO",
            "danger",
            "El cliente presenta señales importantes de posible "
            "abandono. Se recomienda realizar una intervención "
            "preventiva."
        )

    elif probability >= CHURN_THRESHOLD:

        return (
            "🟡 RIESGO MODERADO",
            "warning",
            "El cliente supera el umbral establecido para activar "
            "una estrategia preventiva de retención."
        )

    else:

        return (
            "🟢 RIESGO BAJO",
            "success",
            "El cliente no supera el umbral de abandono definido "
            "para activar una estrategia de retención."
        )


# ==========================================================
# ENCABEZADO
# ==========================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.markdown(
    """
    ### Sistema de análisis y predicción de abandono de clientes

    Plataforma de Business Intelligence que integra **ETL,
    análisis exploratorio, aprendizaje supervisado y visualización
    interactiva** para apoyar la toma de decisiones relacionadas
    con la retención de clientes.
    """
)

st.caption(
    "Proyecto de análisis de datos y Machine Learning"
)


# ==========================================================
# SIDEBAR / NAVEGACIÓN
# ==========================================================

st.sidebar.title(
    "📊 Churn Analytics"
)

st.sidebar.caption(
    "Sistema de análisis y predicción"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🧭 Navegación"
)


# ==========================================================
# MENÚ
# ==========================================================

page = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Resumen ejecutivo",
        "🔎 Análisis de abandono",
        "🔮 Predicción de Churn",
        "🤖 Modelo predictivo",
        "📈 Rendimiento",
        "💡 Decisiones"
    ],
    label_visibility="collapsed"
)


# ==========================================================
# CÁLCULOS GENERALES
# ==========================================================

total_customers = len(df)

churn_customers = (
    df["Churn"] == "Yes"
).sum()

no_churn_customers = (
    df["Churn"] == "No"
).sum()

churn_rate = (
    churn_customers / total_customers
) * 100


# ==========================================================
# PÁGINA 1
# RESUMEN EJECUTIVO
# ==========================================================

if page == "🏠 Resumen ejecutivo":

    st.header(
        "🏠 Resumen ejecutivo"
    )

    st.write(
        """
        Esta sección presenta una visión general del comportamiento
        de los clientes y permite identificar rápidamente la magnitud
        del problema de abandono.
        """
    )


    # ------------------------------------------------------
    # KPIs
    # ------------------------------------------------------

    st.subheader(
        "📌 Indicadores principales"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Total de clientes",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "⚠️ Clientes con Churn",
            f"{churn_customers:,}"
        )


    with col3:

        st.metric(
            "✅ Clientes activos",
            f"{no_churn_customers:,}"
        )


    with col4:

        st.metric(
            "📉 Tasa de abandono",
            f"{churn_rate:.2f}%"
        )


    show_description(
        "¿Qué significan estos indicadores?",
        f"""
        El dataset contiene <strong>{total_customers:,}</strong>
        clientes. De ellos, <strong>{churn_customers:,}</strong>
        abandonaron el servicio, lo que representa una tasa de
        abandono de <strong>{churn_rate:.2f}%</strong>.
        <br><br>
        Estos indicadores permiten dimensionar el problema antes
        de analizar sus posibles causas.
        """
    )


    # ------------------------------------------------------
    # DISTRIBUCIÓN
    # ------------------------------------------------------

    st.subheader(
        "📊 Distribución de clientes"
    )


    churn_data = (
        df["Churn"]
        .value_counts()
        .reset_index()
    )


    churn_data.columns = [
        "Churn",
        "Clientes"
    ]


    churn_data["Estado"] = churn_data[
        "Churn"
    ].map(
        {
            "Yes": "Abandonó",
            "No": "Permanece"
        }
    )


    fig = px.pie(
        churn_data,
        names="Estado",
        values="Clientes",
        hole=0.48,
        title="Distribución general de Churn"
    )


    fig.update_layout(
        height=450,
        legend_title_text="Estado"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Esta gráfica muestra la proporción de clientes que
        permanecieron en el servicio frente a los clientes que
        abandonaron.
        <br><br>
        <strong>Interpretación:</strong> permite observar rápidamente
        la magnitud del problema de Churn y detectar que existe una
        diferencia importante entre ambas clases.
        """
    )


    # ------------------------------------------------------
    # RESUMEN EJECUTIVO
    # ------------------------------------------------------

    st.subheader(
        "📌 Lectura ejecutiva"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div class="danger-card">
            <h3>⚠️ Problema identificado</h3>

            El proyecto identifica una tasa de abandono de
            <strong>{churn_rate:.2f}%</strong>.

            Esto significa que aproximadamente
            <strong>{churn_customers:,}</strong> clientes del
            conjunto analizado presentan Churn.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="success-card">
            <h3>🎯 Objetivo del sistema</h3>

            Utilizar los patrones históricos de los clientes para
            identificar aquellos que presentan mayor probabilidad
            de abandonar y apoyar estrategias de retención.

            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# PÁGINA 2
# ANÁLISIS DE ABANDONO
# ==========================================================

elif page == "🔎 Análisis de abandono":

    st.header(
        "🔎 Análisis de abandono"
    )


    st.write(
        """
        En esta sección se analizan diferentes características de
        los clientes para identificar patrones relacionados con
        el abandono.
        """
    )


    # ------------------------------------------------------
    # FILTROS
    # ------------------------------------------------------

    with st.expander(
        "🎛️ Filtros de análisis",
        expanded=False
    ):

        contract_filter = st.multiselect(
            "Tipo de contrato",
            sorted(
                df["Contract"].unique()
            ),
            default=sorted(
                df["Contract"].unique()
            )
        )


        internet_filter = st.multiselect(
            "Servicio de Internet",
            sorted(
                df["InternetService"].unique()
            ),
            default=sorted(
                df["InternetService"].unique()
            )
        )


    filtered_df = df[
        df["Contract"].isin(contract_filter)
        &
        df["InternetService"].isin(
            internet_filter
        )
    ]


    st.info(
        f"Mostrando {len(filtered_df):,} clientes después de aplicar los filtros."
    )


    # ------------------------------------------------------
    # CONTRATO
    # ------------------------------------------------------

    st.subheader(
        "📄 Churn por tipo de contrato"
    )


    contract_churn = (
        filtered_df.groupby(
            ["Contract", "Churn"]
        )
        .size()
        .reset_index(
            name="Clientes"
        )
    )


    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="Clientes",
        color="Churn",
        barmode="group",
        title="Clientes según tipo de contrato y Churn",
        text_auto=True
    )


    fig_contract.update_layout(
        height=450
    )


    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Compara la cantidad de clientes que permanecen y abandonan
        según el tipo de contrato.
        <br><br>
        <strong>¿Para qué sirve?</strong>
        Permite identificar qué modalidades contractuales concentran
        mayor cantidad de abandonos.
        <br><br>
        <strong>Decisión de negocio:</strong>
        los segmentos con mayor abandono pueden ser candidatos para
        campañas de fidelización, incentivos o migración hacia
        contratos de mayor duración.
        """
    )


    # ------------------------------------------------------
    # INTERNET
    # ------------------------------------------------------

    st.subheader(
        "🌐 Churn por servicio de Internet"
    )


    internet_churn = (
        filtered_df.groupby(
            [
                "InternetService",
                "Churn"
            ]
        )
        .size()
        .reset_index(
            name="Clientes"
        )
    )


    fig_internet = px.bar(
        internet_churn,
        x="InternetService",
        y="Clientes",
        color="Churn",
        barmode="group",
        title="Clientes según servicio de Internet",
        text_auto=True
    )


    fig_internet.update_layout(
        height=450
    )


    st.plotly_chart(
        fig_internet,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Permite comparar el comportamiento de abandono entre los
        diferentes servicios de Internet.
        <br><br>
        <strong>Interpretación:</strong>
        si un servicio presenta una proporción elevada de Churn,
        puede ser necesario investigar factores como precio,
        calidad percibida, soporte técnico o satisfacción.
        """
    )


    # ------------------------------------------------------
    # MÉTODO DE PAGO
    # ------------------------------------------------------

    st.subheader(
        "💳 Churn por método de pago"
    )


    payment_churn = (
        filtered_df.groupby(
            [
                "PaymentMethod",
                "Churn"
            ]
        )
        .size()
        .reset_index(
            name="Clientes"
        )
    )


    fig_payment = px.bar(
        payment_churn,
        x="PaymentMethod",
        y="Clientes",
        color="Churn",
        barmode="group",
        title="Clientes según método de pago",
        text_auto=True
    )


    fig_payment.update_layout(
        height=500,
        xaxis_tickangle=-25
    )


    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Compara el número de clientes que permanecen y abandonan
        según su método de pago.
        <br><br>
        <strong>Valor para negocio:</strong>
        puede ayudar a identificar segmentos que requieren
        comunicación o incentivos específicos.
        """
    )


    # ------------------------------------------------------
    # ANTIGÜEDAD
    # ------------------------------------------------------

    st.subheader(
        "⏳ Antigüedad de los clientes"
    )


    fig_tenure = px.histogram(
        filtered_df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Distribución de clientes según antigüedad",
        marginal="box"
    )


    fig_tenure.update_layout(
        height=500
    )


    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Analiza la distribución de clientes según el número de meses
        que llevan utilizando el servicio.
        <br><br>
        <strong>Interpretación:</strong>
        permite observar si el abandono se concentra especialmente
        entre clientes nuevos o si también existe un comportamiento
        importante en clientes con mayor antigüedad.
        """
    )


    # ------------------------------------------------------
    # CARGOS
    # ------------------------------------------------------

    st.subheader(
        "💰 Cargos mensuales"
    )


    fig_charges = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Distribución de cargos mensuales según Churn",
        points="outliers"
    )


    fig_charges.update_layout(
        height=450
    )


    st.plotly_chart(
        fig_charges,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Compara la distribución de los cargos mensuales entre
        clientes que permanecen y clientes que abandonan.
        <br><br>
        <strong>¿Qué debemos observar?</strong>
        La mediana, dispersión y posibles valores extremos.
        <br><br>
        Una diferencia importante puede indicar que el nivel de
        facturación está relacionado con el comportamiento de
        abandono.
        """
    )


    # ------------------------------------------------------
    # TABLA
    # ------------------------------------------------------

    with st.expander(
        "📋 Ver datos filtrados"
    ):

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )


# ==========================================================
# PÁGINA 3
# PREDICCIÓN
# ==========================================================

elif page == "🔮 Predicción de Churn":

    st.header(
        "🔮 Predicción de abandono"
    )


    st.write(
        """
        Introduce las características de un cliente para calcular
        su probabilidad estimada de abandono.
        """
    )


    show_description(
        "¿Cómo funciona esta sección?",
        """
        El sistema recibe las características del cliente y las
        envía al modelo de Machine Learning.
        <br><br>
        El modelo devuelve una probabilidad de abandono.
        Si esta probabilidad es igual o superior al
        <strong>30%</strong>, el sistema clasifica al cliente como
        potencial Churn.
        """
    )


    # ------------------------------------------------------
    # FORMULARIO
    # ------------------------------------------------------

    with st.form(
        "churn_prediction_form"
    ):

        st.subheader(
            "👤 Información personal"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            gender = st.selectbox(
                "Género",
                [
                    "Female",
                    "Male"
                ]
            )


            senior_citizen = st.selectbox(
                "Adulto mayor",
                [0, 1],
                format_func=lambda x:
                "No" if x == 0 else "Sí"
            )


            partner = st.selectbox(
                "¿Tiene pareja?",
                [
                    "Yes",
                    "No"
                ]
            )


            dependents = st.selectbox(
                "¿Tiene dependientes?",
                [
                    "Yes",
                    "No"
                ]
            )


        with col2:

            phone_service = st.selectbox(
                "Servicio telefónico",
                [
                    "Yes",
                    "No"
                ]
            )


            multiple_lines = st.selectbox(
                "Múltiples líneas",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )


            internet_service = st.selectbox(
                "Servicio de Internet",
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


        st.divider()


        # --------------------------------------------------
        # CONTRATO
        # --------------------------------------------------

        st.subheader(
            "📄 Información contractual"
        )


        col4, col5, col6 = st.columns(3)


        with col4:

            contract = st.selectbox(
                "Tipo de contrato",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )


        with col5:

            paperless_billing = st.selectbox(
                "Facturación electrónica",
                [
                    "Yes",
                    "No"
                ]
            )


        with col6:

            payment_method = st.selectbox(
                "Método de pago",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )


        st.divider()


        # --------------------------------------------------
        # INFORMACIÓN ECONÓMICA
        # --------------------------------------------------

        st.subheader(
            "💰 Información económica"
        )


        col7, col8, col9 = st.columns(3)


        with col7:

            tenure = st.number_input(
                "Antigüedad (meses)",
                min_value=0,
                max_value=72,
                value=12
            )


        with col8:

            monthly_charges = st.number_input(
                "Cargos mensuales",
                min_value=0.0,
                max_value=150.0,
                value=70.0,
                step=0.01
            )


        with col9:

            total_charges = st.number_input(
                "Cargos totales",
                min_value=0.0,
                value=840.0,
                step=0.01
            )


        st.divider()


        submitted = st.form_submit_button(
            "🔮 PREDECIR ABANDONO",
            use_container_width=True
        )


    # ------------------------------------------------------
    # REALIZAR PREDICCIÓN
    # ------------------------------------------------------

    if submitted:

        customer = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )


        try:

            probability = model.predict_proba(
                customer
            )[0][1]

        except Exception as e:

            st.error(
                "❌ No fue posible realizar la predicción."
            )

            st.code(str(e))

            st.stop()


        prediction = (
            "Yes"
            if probability >= CHURN_THRESHOLD
            else "No"
        )


        percentage = probability * 100


        # --------------------------------------------------
        # RESULTADO
        # --------------------------------------------------

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
                f"{CHURN_THRESHOLD:.0%}"
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


        # --------------------------------------------------
        # PROBABILIDAD
        # --------------------------------------------------

        st.subheader(
            "📈 Probabilidad estimada"
        )


        st.progress(
            min(
                max(probability, 0),
                1
            )
        )


        st.caption(
            f"Probabilidad calculada: {percentage:.2f}%"
        )


        # --------------------------------------------------
        # RIESGO
        # --------------------------------------------------

        risk, risk_type, message = risk_information(
            probability
        )


        if risk_type == "danger":

            st.error(
                risk
            )

            st.markdown(
                f"""
**Situación:**  
{message}

**Recomendación:**  
Implementar una estrategia prioritaria de retención.
"""
            )


        elif risk_type == "warning":

            st.warning(
                risk
            )

            st.markdown(
                f"""
**Situación:**  
{message}

**Recomendación:**  
Realizar seguimiento y ofrecer incentivos de retención.
"""
            )


        else:

            st.success(
                risk
            )

            st.markdown(
                f"""
**Situación:**  
{message}

**Recomendación:**  
Mantener el seguimiento habitual.
"""
            )


        # --------------------------------------------------
        # PREDICCIÓN FINAL
        # --------------------------------------------------

        st.divider()


        if prediction == "Yes":

            st.error(
                "⚠️ Predicción final: CLIENTE CON RIESGO DE CHURN"
            )

        else:

            st.success(
                "✅ Predicción final: CLIENTE SIN CHURN"
            )


# ==========================================================
# PÁGINA 4
# MODELO PREDICTIVO
# ==========================================================

elif page == "🤖 Modelo predictivo":

    st.header(
        "🤖 Modelo predictivo"
    )


    st.write(
        """
        Esta sección explica cómo funciona el modelo de Machine
        Learning utilizado para realizar las predicciones.
        """
    )


    # ------------------------------------------------------
    # MODELO
    # ------------------------------------------------------

    st.subheader(
        "🌲 Algoritmo utilizado"
    )


    st.markdown(
        f"""
        <div class="model-card">

        <h2>🌲 {model_name}</h2>

        El proyecto utiliza un algoritmo de aprendizaje supervisado
        para clasificar clientes en dos categorías:

        <br><br>

        🟢 <strong>No Churn</strong> — el cliente permanece.

        <br>

        🔴 <strong>Churn</strong> — el cliente abandona.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------
    # EXPLICACIÓN
    # ------------------------------------------------------

    st.subheader(
        "📚 ¿Cómo funciona Random Forest?"
    )


    st.markdown(
        """
        **Random Forest** es un algoritmo de aprendizaje supervisado
        basado en múltiples árboles de decisión.

        En lugar de depender de un solo árbol, el algoritmo combina
        las predicciones de varios árboles para obtener una decisión
        más robusta.

        En este proyecto, cada árbol analiza diferentes combinaciones
        de las características de los clientes.

        Finalmente, el conjunto de árboles participa en la decisión
        de clasificación del cliente.
        """
    )


    # ------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------

    st.subheader(
        "⚙️ Flujo del modelo"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="model-card">

            ### 1️⃣ Datos

            Dataset limpio con información histórica de clientes.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="model-card">

            ### 2️⃣ Preprocesamiento

            Variables numéricas y categóricas son transformadas
            antes del entrenamiento.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="model-card">

            ### 3️⃣ Entrenamiento

            El modelo aprende patrones relacionados con Churn.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="model-card">

            ### 4️⃣ Predicción

            Se calcula la probabilidad de abandono de un cliente.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------

    st.subheader(
        "🧩 Variables utilizadas"
    )


    predictors = [
        column
        for column in df.columns
        if column not in [
            "customerID",
            "Churn"
        ]
    ]


    st.write(
        f"El modelo utiliza **{len(predictors)} variables predictoras**:"
    )


    cols = st.columns(3)


    for index, variable in enumerate(
        predictors
    ):

        with cols[index % 3]:

            st.markdown(
                f"- `{variable}`"
            )


    # ------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------

    st.subheader(
        "🧪 División de los datos"
    )


    st.markdown(
        """
        Para evaluar el modelo de manera adecuada, los datos se
        dividen en dos conjuntos:

        **80% — entrenamiento**

        Se utiliza para que el algoritmo aprenda los patrones.

        **20% — prueba**

        Se utiliza para comprobar qué tan bien funciona el modelo
        con datos que no utilizó durante el entrenamiento.
        """
    )


    # ------------------------------------------------------
    # UMBRAL
    # ------------------------------------------------------

    st.subheader(
        "🎯 Umbral de decisión"
    )


    st.metric(
        "Umbral utilizado",
        f"{CHURN_THRESHOLD:.0%}"
    )


    st.markdown(
        f"""
        El modelo produce una probabilidad entre 0% y 100%.

        En este proyecto se utiliza un umbral de
        **{CHURN_THRESHOLD:.0%}**.

        Por ejemplo:

        - Si la probabilidad es **20%** → No Churn.
        - Si la probabilidad es **30%** → Churn.
        - Si la probabilidad es **70%** → Churn.

        El objetivo de utilizar un umbral relativamente bajo es
        detectar una mayor cantidad de clientes potencialmente
        propensos al abandono.
        """
    )


# ==========================================================
# PÁGINA 5
# RENDIMIENTO
# ==========================================================

elif page == "📈 Rendimiento":

    st.header(
        "📈 Rendimiento del modelo"
    )


    st.write(
        """
        Esta sección presenta las métricas utilizadas para evaluar
        el desempeño del modelo de clasificación.
        """
    )


    # ------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------

    st.subheader(
        "📊 Métricas de evaluación"
    )


    # Resultados obtenidos anteriormente para Random Forest
    accuracy = 0.8070
    precision = 0.6821
    recall = 0.5107
    f1 = 0.5841


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision:.2%}"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall:.2%}"
        )


    with col4:

        st.metric(
            "F1-Score",
            f"{f1:.2%}"
        )


    show_description(
        "¿Cómo interpretamos estas métricas?",
        f"""
        <strong>Accuracy ({accuracy:.2%})</strong>:
        proporción total de predicciones correctas.
        <br><br>

        <strong>Precision ({precision:.2%})</strong>:
        de los clientes clasificados como Churn, indica qué proporción
        realmente abandonó.
        <br><br>

        <strong>Recall ({recall:.2%})</strong>:
        indica qué proporción de los clientes que realmente abandonaron
        logró detectar el modelo.
        <br><br>

        <strong>F1-Score ({f1:.2%})</strong>:
        representa el equilibrio entre Precision y Recall.
        """
    )


    # ------------------------------------------------------
    # COMPARACIÓN
    # ------------------------------------------------------

    st.subheader(
        "🏆 Comparación de modelos"
    )


    comparison = pd.DataFrame(
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

            "F1-Score": [
                0.6040,
                0.5989,
                0.5841
            ]
        }
    )


    metric_to_show = st.selectbox(
        "Métrica a comparar",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]
    )


    fig_comparison = px.bar(
        comparison,
        x="Modelo",
        y=metric_to_show,
        text_auto=".2%",
        title=f"Comparación de modelos — {metric_to_show}"
    )


    fig_comparison.update_layout(
        yaxis_tickformat=".0%",
        height=450
    )


    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )


    show_description(
        "¿Qué muestra esta gráfica?",
        """
        Permite comparar el desempeño de tres algoritmos supervisados
        probados durante el proyecto.
        <br><br>
        La selección del modelo no debe basarse únicamente en Accuracy.
        Para un problema de Churn también es importante analizar Recall,
        ya que representa la capacidad de detectar clientes que realmente
        abandonarán.
        """
    )


    # ------------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    # ------------------------------------------------------

    st.subheader(
        "🧮 Matriz de confusión"
    )


    confusion = pd.DataFrame(
        [
            [946, 89],
            [183, 191]
        ],
        index=[
            "Real: No Churn",
            "Real: Churn"
        ],
        columns=[
            "Predicción: No Churn",
            "Predicción: Churn"
        ]
    )


    fig_matrix = px.imshow(
        confusion,
        text_auto=True,
        aspect="auto",
        title="Matriz de confusión — Random Forest"
    )


    fig_matrix.update_layout(
        height=450
    )


    st.plotly_chart(
        fig_matrix,
        use_container_width=True
    )


    show_description(
        "¿Cómo se interpreta?",
        """
        La matriz de confusión permite observar qué tan bien clasifica
        el modelo cada categoría.
        <br><br>

        <strong>946</strong> clientes fueron correctamente clasificados
        como No Churn.
        <br>

        <strong>191</strong> clientes fueron correctamente clasificados
        como Churn.
        <br><br>

        También existen falsos positivos y falsos negativos.
        Para un sistema de retención, los falsos negativos son
        especialmente importantes porque representan clientes que
        abandonaron pero que el modelo no logró detectar.
        """
    )


    # ------------------------------------------------------
    # UMBRAL
    # ------------------------------------------------------

    st.subheader(
        "🎯 Optimización del umbral"
    )


    threshold_data = pd.DataFrame(
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

            "F1-Score": [
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


    threshold_metric = st.selectbox(
        "Métrica",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],
        key="threshold_metric"
    )


    fig_threshold = px.line(
        threshold_data,
        x="Umbral",
        y=threshold_metric,
        markers=True,
        title=f"Comportamiento de {threshold_metric} según el umbral"
    )


    fig_threshold.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        height=450
    )


    st.plotly_chart(
        fig_threshold,
        use_container_width=True
    )


    show_description(
        "¿Por qué es importante el umbral?",
        """
        El umbral determina a partir de qué probabilidad el sistema
        considera que un cliente presenta Churn.
        <br><br>

        Un umbral menor permite detectar más posibles abandonos,
        aumentando el Recall, pero también genera más falsos positivos.
        <br><br>

        En este proyecto se seleccionó un umbral del
        <strong>30%</strong> como estrategia orientada a detectar una
        mayor cantidad de clientes potencialmente en riesgo.
        """
    )


# ==========================================================
# PÁGINA 6
# DECISIONES
# ==========================================================

elif page == "💡 Decisiones":

    st.header(
        "💡 Insights y toma de decisiones"
    )


    st.write(
        """
        El objetivo final del análisis no es solamente generar
        gráficas, sino convertir los resultados obtenidos en
        acciones que puedan apoyar la toma de decisiones.
        """
    )


    # ------------------------------------------------------
    # HALLAZGO 1
    # ------------------------------------------------------

    st.subheader(
        "1️⃣ El abandono es un problema relevante"
    )


    st.markdown(
        f"""
        La tasa general de abandono es de
        **{churn_rate:.2f}%**.

        Esto significa que existe un segmento importante de clientes
        que no permanece en el servicio.
        """
    )


    # ------------------------------------------------------
    # HALLAZGO 2
    # ------------------------------------------------------

    st.subheader(
        "2️⃣ El contrato es una variable importante"
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


    for contract, rate in contract_rates.items():

        st.write(
            f"**{contract}:** {rate:.2f}% de abandono"
        )


    show_description(
        "Decisión recomendada",
        """
        Analizar estrategias específicas para los clientes que
        presentan mayor propensión al abandono según su modalidad
        contractual.
        <br><br>
        Algunas estrategias pueden incluir incentivos, descuentos,
        beneficios por permanencia o migración hacia contratos de
        mayor duración.
        """
    )


    # ------------------------------------------------------
    # HALLAZGO 3
    # ------------------------------------------------------

    st.subheader(
        "3️⃣ Utilizar el modelo para priorizar clientes"
    )


    st.markdown(
        """
        El modelo permite pasar de un análisis descriptivo a un
        enfoque predictivo.

        En lugar de esperar a que el cliente abandone, la empresa
        puede identificar clientes con mayor probabilidad de Churn
        y priorizar acciones preventivas.
        """
    )


    # ------------------------------------------------------
    # MATRIZ DE ACCIONES
    # ------------------------------------------------------

    st.subheader(
        "🎯 Matriz de acciones recomendadas"
    )


    actions = pd.DataFrame(
        {
            "Nivel": [
                "🟢 Bajo",
                "🟡 Moderado",
                "🟠 Alto",
                "🔴 Muy alto"
            ],

            "Probabilidad": [
                "< 30%",
                "30% - 49%",
                "50% - 69%",
                "≥ 70%"
            ],

            "Acción": [
                "Seguimiento habitual",
                "Contacto preventivo",
                "Oferta de retención",
                "Intervención prioritaria"
            ]
        }
    )


    st.dataframe(
        actions,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # CONCLUSIÓN
    # ------------------------------------------------------

    st.subheader(
        "🏁 Conclusión"
    )


    st.markdown(
        """
        El sistema integra un proceso completo de extracción,
        transformación, análisis, modelado y visualización.

        La información histórica permite identificar patrones
        asociados con el abandono y el modelo supervisado permite
        estimar el riesgo individual de cada cliente.

        De esta manera, el proyecto transforma los datos en
        información útil para apoyar estrategias de retención y
        toma de decisiones.
        """
    )


    st.markdown(
        """
        <div class="success-card">
        <h3>🎯 Objetivo final</h3>

        Detectar clientes potencialmente propensos al abandono
        <strong>antes de que abandonen</strong>, permitiendo que
        la empresa pueda actuar de manera preventiva.

        </div>
        """,
        unsafe_allow_html=True
    )