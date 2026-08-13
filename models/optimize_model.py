import pandas as pd
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
    confusion_matrix
)

# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

# CARGAR DATASET
def load_data():

    print("=" * 60)
    print("OPTIMIZACIÓN DEL MODELO DE CHURN")
    print("=" * 60)

    print("\nArchivo utilizado:")
    print(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    print("\n✓ Dataset cargado correctamente")

    print(f"Registros: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    return df

# PREPARAR VARIABLES
def prepare_data(df):

    print("\n" + "=" * 60)
    print("PREPARACIÓN DE VARIABLES")
    print("=" * 60)

    # Variable objetivo
    y = df["Churn"]

    # Variables predictoras
    X = df.drop(
        columns=["Churn", "customerID"]
    )

    print("\nVariable objetivo:")
    print("Churn")

    print("\nVariables predictoras:")
    print(f"Cantidad: {X.shape[1]}")

    return X, y

# CREAR PREPROCESADOR
def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

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

    return preprocessor


# ==========================================
# CREAR MODELO
# ==========================================

def create_model(preprocessor):

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

    return model


# ==========================================
# DIVISIÓN DE DATOS
# ==========================================

def split_data(X, y):

    print("\n" + "=" * 60)
    print("DIVISIÓN TRAIN / TEST")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nEntrenamiento:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nPrueba:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ==========================================
# ENTRENAMIENTO
# ==========================================

def train_model(model, X_train, y_train):

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO")
    print("=" * 60)

    print("\nEntrenando Regresión Logística...")

    model.fit(
        X_train,
        y_train
    )

    print("\n✓ Modelo entrenado correctamente.")

    return model


# ==========================================
# OPTIMIZAR UMBRAL
# ==========================================

def evaluate_thresholds(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("EVALUACIÓN DE DIFERENTES UMBRALES")
    print("=" * 60)

    # ------------------------------------------
    # Probabilidad de Churn
    # ------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    results = []

    print(
        "\n"
        f"{'Umbral':<10}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 58)

    for threshold in thresholds:

        # Convertir probabilidad en clase
        y_pred = (
            probabilities >= threshold
        ).astype(int)

        # Convertir valores reales
        y_test_binary = (
            y_test == "Yes"
        ).astype(int)

        # Métricas
        accuracy = accuracy_score(
            y_test_binary,
            y_pred
        )

        precision = precision_score(
            y_test_binary,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test_binary,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test_binary,
            y_pred,
            zero_division=0
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"{threshold:<10.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    return pd.DataFrame(results)


# ==========================================
# SELECCIONAR MEJOR UMBRAL
# ==========================================

def select_best_threshold(results):

    print("\n" + "=" * 60)
    print("SELECCIÓN DEL UMBRAL")
    print("=" * 60)

    # Seleccionamos el umbral con mejor F1
    best_row = results.loc[
        results["f1"].idxmax()
    ]

    threshold = best_row["threshold"]

    print("\nMejor umbral según F1-Score:")

    print(
        f"Umbral: {threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1-Score: "
        f"{best_row['f1']:.4f}"
    )

    return threshold


# ==========================================
# EVALUACIÓN FINAL
# ==========================================

def final_evaluation(
    model,
    X_test,
    y_test,
    threshold
):

    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL DEL UMBRAL")
    print("=" * 60)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    y_test_binary = (
        y_test == "Yes"
    ).astype(int)

    accuracy = accuracy_score(
        y_test_binary,
        y_pred
    )

    precision = precision_score(
        y_test_binary,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test_binary,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test_binary,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test_binary,
        y_pred
    )

    print(
        f"\nUmbral utilizado: {threshold:.2f}"
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
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

    print("\nMatriz de confusión:")

    print(matrix)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

if __name__ == "__main__":

    # 1. Cargar datos
    df = load_data()

    # 2. Preparar variables
    X, y = prepare_data(df)

    # 3. Crear preprocesador
    preprocessor = create_preprocessor(X)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # 5. Crear modelo
    model = create_model(
        preprocessor
    )

    # 6. Entrenar
    model = train_model(
        model,
        X_train,
        y_train
    )

    # 7. Evaluar umbrales
    results = evaluate_thresholds(
        model,
        X_test,
        y_test
    )

    # 8. Seleccionar mejor umbral
    best_threshold = select_best_threshold(
        results
    )

    # 9. Evaluación final
    final_metrics = final_evaluation(
        model,
        X_test,
        y_test,
        best_threshold
    )

    print("\n" + "=" * 60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("=" * 60)

    print(
        "\n✓ Se evaluaron diferentes umbrales."
    )

    print(
        f"✓ Umbral seleccionado: "
        f"{best_threshold:.2f}"
    )