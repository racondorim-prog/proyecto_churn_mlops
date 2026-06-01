from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_inicio():
    response = client.get("/")

    assert response.status_code == 200
    assert "mensaje" in response.json()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert "estado" in response.json()
    assert "modelo_disponible" in response.json()


def test_prediccion_cliente_bajo_riesgo():
    payload = {
        "edad": 30,
        "antiguedad_meses": 24,
        "saldo_promedio": 1500.0,
        "reclamos": 0,
        "usa_app": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "churn_predicho" in data
    assert "probabilidad_churn" in data
    assert data["churn_predicho"] in [0, 1]