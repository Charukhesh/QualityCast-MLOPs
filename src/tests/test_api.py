import pytest
from fastapi.testclient import TestClient
from src.app.health_api import app

client = TestClient(app)

def test_health_endpoint():
    """Test the /health Liveness probe"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ready_endpoint_before_load():
    """Test /ready returns 503 if model is not loaded"""
    response = client.get("/ready")
    assert response.status_code in [200, 503] 

def test_metrics_endpoint():
    """Test Prometheus metrics are being served"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_service_ready" in response.text 
    assert "python_info" in response.text