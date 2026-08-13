import pandas as pd
import joblib

from pathlib import Path

# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "churn_model.pkl"
)

CHURN_THRESHOLD = 0.30

# CARGAR MODELO
def load_model():

    print("=" * 60)
    print("CARGANDO MODELO DE PREDICCIÓN")
    print("=" * 60)

    print("\nArchivo utilizado:")
    print(MODEL_FILE)

    model = joblib.load(MODEL_FILE)

    print("\n✓ Modelo cargado correctamente.")

    return model

# CREAR CLIENTE DE PRUEBA
def create_customer():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.00,
        "TotalCharges": 140.00
    }

    return pd.DataFrame([customer])

# REALIZAR PREDICCIÓN
def predict_churn(model, customer):

    print("\n" + "=" * 60)
    print("REALIZANDO PREDICCIÓN")
    print("=" * 60)

    # Obtener probabilidad de Churn
    probability = model.predict_proba(
        customer
    )[0][1]

    # Aplicar umbral
    prediction = (
        "Yes"
        if probability >= CHURN_THRESHOLD
        else "No"
    )

    return probability, prediction

# MOSTRAR RESULTADO
def show_result(probability, prediction):

    print("\n" + "=" * 60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)

    print(
        f"\nProbabilidad de abandono: "
        f"{probability:.2%}"
    )

    print(
        f"Umbral utilizado: "
        f"{CHURN_THRESHOLD:.0%}"
    )

    print(
        f"\nPredicción: "
        f"{prediction}"
    )

    if prediction == "Yes":

        print("\n⚠️ CLIENTE EN RIESGO DE ABANDONO")

        print(
            "Se recomienda implementar "
            "una estrategia de retención."
        )

    else:

        print("\n✓ CLIENTE SIN RIESGO ALTO DE ABANDONO")

        print(
            "No se requiere una acción inmediata."
        )

# PROGRAMA PRINCIPAL
if __name__ == "__main__":

    # 1. Cargar modelo
    model = load_model()

    # 2. Crear cliente
    customer = create_customer()

    print("\nDatos del cliente:")

    print(customer.to_string(
        index=False
    ))

    # 3. Realizar predicción
    probability, prediction = predict_churn(
        model,
        customer
    )

    # 4. Mostrar resultado
    show_result(
        probability,
        prediction
    )

    print("\n" + "=" * 60)
    print("PREDICCIÓN COMPLETADA")
    print("=" * 60)