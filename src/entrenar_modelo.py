from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_DATA = DATA_DIR / "train.csv"

MODEL_FILE_LOGISTIC = MODELS_DIR / "modelo_churn.pkl"
MODEL_FILE_TREE = MODELS_DIR / "modelo_churn_arbol.pkl"


def entrenar_modelo():
    """
    Entrena dos modelos de clasificación para predecir churn.

    Experimentos realizados:
    1. Modificación de hiperparámetro en Regresión Logística: max_iter=500.
    2. Entrenamiento de un segundo algoritmo: Árbol de Decisión.
    """

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preparar_datos.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    modelo_logistico = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression(max_iter=500)),
        ]
    )

    modelo_arbol = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    modelo_logistico.fit(X, y)
    modelo_arbol.fit(X, y)

    joblib.dump(modelo_logistico, MODEL_FILE_LOGISTIC)
    joblib.dump(modelo_arbol, MODEL_FILE_TREE)

    print("Modelos entrenados correctamente.")
    print(f"Modelo de Regresión Logística guardado en: {MODEL_FILE_LOGISTIC}")
    print(f"Modelo de Árbol de Decisión guardado en: {MODEL_FILE_TREE}")


if __name__ == "__main__":
    entrenar_modelo()