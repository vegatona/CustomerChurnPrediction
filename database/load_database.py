import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

DATABASE_FILE = (
    BASE_DIR
    / "database"
    / "customer_churn.db"
)

# Crear conexión SQLite
engine = create_engine(
    f"sqlite:///{DATABASE_FILE}"
)

# CARGA
print("=" * 50)
print("CARGA A SQLITE")
print("=" * 50)

print("\nLeyendo dataset limpio...")

df = pd.read_csv(INPUT_FILE)

print(f"Registros: {len(df)}")

print("\nGuardando en SQLite...")

df.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

print("\n✓ Datos cargados correctamente.")

print("\nBase creada:")

print(DATABASE_FILE)