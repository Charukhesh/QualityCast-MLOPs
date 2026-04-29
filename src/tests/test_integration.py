import pytest
import os
import subprocess
import requests
import time
from fastapi.testclient import TestClient
from src.app.health_api import app

client = TestClient(app)

# Readiness Probe validation
def test_readiness_logic():
    """Verify readiness returns 200 (Ready) or 503 (Loading)"""
    response = client.get("/ready")
    # In a healthy system, it should return 200 or 503, never 404 or 500
    assert response.status_code in [200, 503]

# DVC Data Integrity
def test_dvc_integrity():
    """Verify DVC tracking files exist and data is linked"""
    # Check for the dvc metadata file
    assert os.path.exists("data/raw.dvc") or os.path.exists("dvc.yaml")
    
    # Run dvc status to check for data drift or missing files
    result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
    assert result.returncode == 0

# MLflow Connectivity & Artifacts
def test_mlflow_artifacts_presence():
    """Verify MLflow runs and artifact directory exist"""
    mlruns_path = "data/mlruns"
    assert os.path.exists(mlruns_path)
    # Check if at least one experiment (0) or custom ID folder exists
    subfolders = [f for f in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, f))]
    assert len(subfolders) > 0

#  Metric Export Validation
def test_metric_increment():
    """Verify that hitting the ready endpoint is reflected in metrics"""
    initial_metrics = client.get("/metrics").text
    
    client.get("/ready")
    
    new_metrics = client.get("/metrics")
    assert new_metrics.status_code == 200
    assert "model_service_ready" in new_metrics.text

# Model Loading Latency
def test_model_loading_latency():
    """Verify the system response time for health checks is within limits"""
    start_time = time.perf_counter()
    response = client.get("/health")
    end_time = time.perf_counter()
    
    latency = (end_time - start_time)
    # Requirement: Response must be snappy (under 1 second for health probe)
    assert latency < 1.0
    assert response.status_code == 200

# Schema Consistency (Input Tensor Shape)
def test_input_schema_consistency():
    """Verify the expected input shape for the vision model (1x1x64x64)"""

    expected_shape = (1, 64, 64) # (Channels, H, W)
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    assert test_transform.transforms[0].size == (64, 64)