import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "churn_model.pkl"
)

# Umbral seleccionado después de la optimización
CHURN_THRESHOLD = 0.30

# CARGAR DATASET
def load_data():

    print("=" * 50)
    print("CARGANDO DATASET PARA MACHINE LEARNING")
    print("=" * 50)

    print("\nArchivo utilizado:")
    print(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    print("\n✓ Dataset cargado correctamente")

    print(f"Registros: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    return df

# SEPARAR VARIABLES
def prepare_features(df):

    print("\n" + "=" * 50)
    print("PREPARACIÓN DE VARIABLES")
    print("=" * 50)

    # VARIABLE OBJETIVO
    y = df["Churn"]

    # VARIABLES PREDICTORAS
    X = df.drop(
        columns=[
            "Churn",
            "customerID"
        ]
    )

    print("\nVariable objetivo:")
    print("Churn")

    print("\nVariables predictoras:")
    print(f"Cantidad: {X.shape[1]}")

    for column in X.columns:
        print(f"- {column}")

    return X, y

# PREPROCESAMIENTO
def create_preprocessor(X):

    print("\n" + "=" * 50)
    print("PREPROCESAMIENTO")
    print("=" * 50)

    # VARIABLES NUMÉRICAS
    numeric_features = X.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    ).columns.tolist()

    # VARIABLES CATEGÓRICAS
    categorical_features = X.select_dtypes(
        include=[
            "object",
            "string"
        ]
    ).columns.tolist()

    print("\nVariables numéricas:")

    for column in numeric_features:
        print(f"- {column}")

    print(
        f"\nTotal numéricas: "
        f"{len(numeric_features)}"
    )

    print("\nVariables categóricas:")

    for column in categorical_features:
        print(f"- {column}")

    print(
        f"\nTotal categóricas: "
        f"{len(categorical_features)}"
    )

    # COLUMN TRANSFORMER
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    print(
        "\n✓ Preprocesador creado correctamente."
    )

    print(
        "✓ Variables numéricas serán escaladas."
    )

    print(
        "✓ Variables categóricas serán codificadas."
    )

    return preprocessor

# DIVISIÓN TRAIN / TEST
def split_data(X, y):

    print("\n" + "=" * 50)
    print("DIVISIÓN TRAIN / TEST")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDatos de entrenamiento:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nDatos de prueba:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    print("\n✓ División completada.")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

# CREAR MODELO FINAL
def create_model(preprocessor):

    print("\n" + "=" * 50)
    print("CREANDO MODELO FINAL")
    print("=" * 50)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    print("\nModelo seleccionado:")
    print("Regresión Logística")

    print("\nConfiguración:")
    print("- max_iter: 2000")
    print("- random_state: 42")

    print("\nUmbral de clasificación:")
    print(f"{CHURN_THRESHOLD}")

    print("\n✓ Pipeline creado correctamente.")

    return model

# ENTRENAR MODELO
def train_model(
    model,
    X_train,
    y_train
):

    print("\n" + "=" * 50)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 50)

    print("\nEntrenando...")

    model.fit(
        X_train,
        y_train
    )

    print("\n✓ Modelo entrenado correctamente.")

    return model

# EVALUAR MODELO CON UMBRAL
def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 50)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 50)

    # PROBABILIDADES
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # APLICAR UMBRAL
    y_pred_binary = (
        probabilities >= CHURN_THRESHOLD
    ).astype(int)

    # Convertir a etiquetas originales
    y_pred = pd.Series(
        y_pred_binary,
        index=y_test.index
    ).map({
        0: "No",
        1: "Yes"
    })

    # MÉTRICAS
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    # MOSTRAR MÉTRICAS
    print("\n--- CONFIGURACIÓN ---")

    print(
        f"Umbral utilizado: "
        f"{CHURN_THRESHOLD:.2f}"
    )

    print("\n--- MÉTRICAS ---")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    # MATRIZ DE CONFUSIÓN
    print("\n--- MATRIZ DE CONFUSIÓN ---")

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "No",
            "Yes"
        ]
    )

    print(matrix)

    # REPORTE
    print("\n--- REPORTE DE CLASIFICACIÓN ---")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# GUARDAR MODELO
def save_model(model):

    print("\n" + "=" * 50)
    print("GUARDANDO MODELO")
    print("=" * 50)

    # Crear carpeta si no existe
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar modelo
    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\n✓ Modelo guardado correctamente.")

    print("\nArchivo:")
    print(MODEL_FILE)

    print("\nUmbral asociado:")
    print(CHURN_THRESHOLD)

# PROGRAMA PRINCIPAL
if __name__ == "__main__":

    # 1. CARGAR DATOS
    df = load_data()

    # 2. PREPARAR VARIABLES
    X, y = prepare_features(df)

    # 3. CREAR PREPROCESADOR
    preprocessor = create_preprocessor(
        X
    )

    # 4. DIVIDIR DATOS
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # 5. CREAR MODELO
    model = create_model(
        preprocessor
    )

    # 6. ENTRENAR
    model = train_model(
        model,
        X_train,
        y_train
    )

    # 7. EVALUAR
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # 8. GUARDAR MODELO
    save_model(
        model
    )

    # FINAL
    print("\n" + "=" * 50)
    print("PROCESO COMPLETADO")
    print("=" * 50)

    print("\n✓ Regresión Logística entrenada.")

    print(
        f"✓ Umbral seleccionado: "
        f"{CHURN_THRESHOLD:.2f}"
    )

    print("✓ Evaluación completada.")

    print("✓ Modelo guardado correctamente.")