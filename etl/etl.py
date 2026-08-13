import pandas as pd
from pathlib import Path

# CONFIGURACIÓN DEL PROYECTO
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = (
    BASE_DIR
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

# EXTRACCIÓN
def extract_data():
    """
    Extrae el dataset original desde un archivo CSV.
    """

    print("=" * 50)
    print("ETAPA DE EXTRACCIÓN")
    print("=" * 50)

    print("\nArchivo utilizado:")
    print(RAW_DATA)

    # Leer el archivo CSV
    df = pd.read_csv(RAW_DATA)

    print("\n✓ Dataset cargado correctamente")

    print(f"\nNúmero de registros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")

    return df

# DIAGNÓSTICO DE DATOS
def diagnose_data(df):
    """
    Realiza un diagnóstico inicial de la calidad
    y estructura del dataset.
    """

    print("\n" + "=" * 50)
    print("DIAGNÓSTICO DEL DATASET")
    print("=" * 50)

    # Información general
    print("\n--- INFORMACIÓN GENERAL ---")

    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    # Tipos de datos
    print("\n--- TIPOS DE DATOS ---")

    print(df.dtypes)

    # Valores nulos
    print("\n--- VALORES NULOS ---")

    nulls = df.isnull().sum()

    if nulls.sum() == 0:
        print("No se encontraron valores nulos.")
    else:
        print(nulls[nulls > 0])

    # Cadenas vacías
    print("\n--- CADENAS VACÍAS ---")

    empty_values = (df == "").sum()

    if empty_values.sum() == 0:
        print("No se encontraron cadenas vacías.")
    else:
        print(empty_values[empty_values > 0])

    # Duplicados
    print("\n--- DUPLICADOS ---")

    duplicates = df.duplicated().sum()

    print(f"Registros duplicados: {duplicates}")

    # Valores únicos
    print("\n--- VALORES ÚNICOS ---")

    for column in df.columns:

        unique_values = df[column].nunique()

        print(f"{column}: {unique_values}")

    # Distribución de Churn
    print("\n--- DISTRIBUCIÓN DE CHURN ---")

    churn_counts = df["Churn"].value_counts()

    print(churn_counts)

    print("\nPorcentaje:")

    churn_percentage = (
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(churn_percentage)

    # Estadísticas numéricas
    print("\n--- ESTADÍSTICAS NUMÉRICAS ---")

    print(df.describe())

# DIAGNÓSTICO DE TOTALCHARGES
def diagnose_total_charges(df):
    """
    Identifica valores de TotalCharges que no pueden
    convertirse correctamente a números.
    """

    print("\n" + "=" * 50)
    print("DIAGNÓSTICO DE TOTALCHARGES")
    print("=" * 50)

    # Intentar convertir TotalCharges a número.
    # Los valores que no puedan convertirse serán NaN.
    total_charges_numeric = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Identificar registros problemáticos
    invalid_total_charges = df[
        total_charges_numeric.isna()
    ]

    print(
        f"\nRegistros problemáticos: "
        f"{len(invalid_total_charges)}"
    )

    if len(invalid_total_charges) > 0:

        print("\nRegistros encontrados:")

        print(
            invalid_total_charges[
                [
                    "customerID",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "Churn"
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "No se encontraron valores problemáticos "
            "en TotalCharges."
        )

# LIMPIEZA DE DATOS
def clean_data(df):
    """
    Limpia y transforma los datos del dataset.
    """

    print("\n" + "=" * 50)
    print("ETAPA DE LIMPIEZA")
    print("=" * 50)

    # Convertir TotalCharges a numérico
    print("\nConvirtiendo TotalCharges a tipo numérico...")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Detectar valores faltantes
    missing_mask = df["TotalCharges"].isna()

    missing_count = missing_mask.sum()

    print(
        f"Valores faltantes detectados en "
        f"TotalCharges: {missing_count}"
    )

    # Verificar la causa de los valores faltantes
    if missing_count > 0:

        invalid_tenure = df[
            missing_mask & (df["tenure"] != 0)
        ]

        if len(invalid_tenure) > 0:

            print(
                "\n⚠ ADVERTENCIA:"
                "\nExisten valores faltantes en "
                "TotalCharges con tenure diferente de 0."
            )

            print(
                invalid_tenure[
                    [
                        "customerID",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges"
                    ]
                ]
            )

            raise ValueError(
                "No se puede completar TotalCharges "
                "automáticamente."
            )

        # Reemplazar únicamente los casos donde
        # tenure = 0
        print(
            "\nLos valores faltantes corresponden "
            "a clientes con tenure = 0."
        )

        print(
            "Se reemplazarán TotalCharges faltantes "
            "por 0."
        )

        df.loc[missing_mask, "TotalCharges"] = 0

    # Verificación final de la limpieza
    remaining_missing = df["TotalCharges"].isna().sum()

    print(
        f"\nValores faltantes restantes: "
        f"{remaining_missing}"
    )

    print(
        f"Tipo de dato de TotalCharges: "
        f"{df['TotalCharges'].dtype}"
    )

    print("\n✓ Limpieza completada.")

    return df

# VALIDACIÓN DE DATOS LIMPIOS
def validate_data(df):
    """
    Valida que el dataset cumpla con las condiciones
    esperadas después de la limpieza.
    """

    print("\n" + "=" * 50)
    print("VALIDACIÓN DE DATOS")
    print("=" * 50)

    # Cantidad de registros y columnas
    print("\n--- DIMENSIONES ---")

    print(f"Registros: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    # Valores nulos
    print("\n--- VALORES NULOS ---")

    total_nulls = df.isnull().sum().sum()

    print(f"Valores nulos totales: {total_nulls}")

    # Duplicados
    print("\n--- DUPLICADOS ---")

    duplicates = df.duplicated().sum()

    print(f"Registros duplicados: {duplicates}")

    # customerID
    print("\n--- CUSTOMER ID ---")

    unique_customers = df["customerID"].nunique()

    print(f"Clientes únicos: {unique_customers}")

    # TotalCharges
    print("\n--- TOTALCHARGES ---")

    print(
        f"Tipo de dato: "
        f"{df['TotalCharges'].dtype}"
    )

    print(
        f"Valores faltantes: "
        f"{df['TotalCharges'].isna().sum()}"
    )

    # Churn
    print("\n--- CHURN ---")

    print(
        f"Valores únicos: "
        f"{df['Churn'].unique().tolist()}"
    )

    # Resultado final
    if (
        df.shape[0] == 7043
        and df.shape[1] == 21
        and total_nulls == 0
        and duplicates == 0
        and unique_customers == 7043
        and df["TotalCharges"].dtype == "float64"
        and set(df["Churn"].unique()) == {"Yes", "No"}
    ):

        print("\n✓ VALIDACIÓN EXITOSA")
        print("El dataset está listo para ser guardado.")

        return True

    else:

        print("\n✗ VALIDACIÓN FALLIDA")
        print("Revisar los datos antes de continuar.")

        return False

# GUARDAR DATOS PROCESADOS
def save_data(df):
    """
    Guarda el dataset limpio en la carpeta processed.
    """

    processed_dir = (
        BASE_DIR
        / "data"
        / "processed"
    )

    # Crear carpeta si no existe
    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        processed_dir
        / "telco_customer_churn_clean.csv"
    )

    print("\n" + "=" * 50)
    print("GUARDANDO DATASET PROCESADO")
    print("=" * 50)

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\n✓ Dataset guardado correctamente:"
    )

    print(output_file)

    return output_file

# PROGRAMA PRINCIPAL
if __name__ == "__main__":

    # 1. EXTRACCIÓN
    df = extract_data()

    # 2. PRIMEROS REGISTROS
    print("\n" + "=" * 50)
    print("PRIMEROS 5 REGISTROS")
    print("=" * 50)

    print(df.head())

    # 3. COLUMNAS
    print("\n" + "=" * 50)
    print("COLUMNAS DEL DATASET")
    print("=" * 50)

    for columna in df.columns:
        print(f"- {columna}")

    # 4. DIAGNÓSTICO GENERAL
    diagnose_data(df)

    # 5. DIAGNÓSTICO TOTALCHARGES
    diagnose_total_charges(df)

    # 6. LIMPIEZA
    df = clean_data(df)

    # 7. VALIDACIÓN
    is_valid = validate_data(df)

    # 8. GUARDAR DATOS
    if is_valid:
        save_data(df)