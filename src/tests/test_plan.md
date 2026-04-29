# 🛡️ QualityCast-MLOps: Comprehensive Test Plan (v1.2.0)

## 1. Overview
This document outlines the testing strategy for the QualityCast MLOps ecosystem. The goal is to ensure system stability, data provenance, and model reliability across the "No-Cloud" industrial infrastructure.

## 2. Testing Hierarchy

### 2.1 Level 1: Unit Testing (Pytest)
Focuses on the atomic logic of the application and API connectivity.
- **API Functional Tests**: Validation of `/health`, `/ready`, and `/metrics` response codes.
- **Logic Validation**: Ensuring image preprocessing (Resizing/Grayscale) maintains tensor integrity.
- **Schema Validation**: Verifying that the vision model input meets the strict $1 \times 1 \times 64 \times 64$ requirement.

### 2.2 Level 2: Integration Testing (Infrastructure Handshake)
Validates the communication between containerized microservices.
- **DVC Integrity**: Ensures the data pipeline is "locked" and metadata matches the physical storage.
- **MLflow Connectivity**: Validates the handshake between the Airflow worker and the PostgreSQL experiment tracking backend.
- **Prometheus Telemetry**: Confirms the scraper can successfully access and parse the FastAPI metrics endpoint.

### 2.3 Level 3: System Validation (End-to-End)
Validates the full DAG execution flow from raw data ingestion to dashboard population.

## 3. Formal System Validation Matrix

| Test ID | Title | Methodology | Success Criteria |
| :--- | :--- | :--- | :--- |
| **TC-01** | Liveness Check | `GET /health` | Status 200; `{"status": "healthy"}` |
| **TC-02** | Readiness Check | `GET /ready` | Status 503 during load; 200 when ready |
| **TC-03** | DVC Integrity | `dvc status` | No modified dependencies or missing files |
| **TC-04** | MLflow Persistence | Check `data/mlruns` | Artifacts (.pth) present in specific Run ID |
| **TC-05** | Metric Export | `GET /metrics` | Counter `ok_total` increments on prediction |
| **TC-06** | Performance Latency | Timer decorator | Inference response < 200ms |
| **TC-07** | Image Schema | `torch.shape` | Output tensor must be [1, 1, 64, 64] |

## 4. Operational Acceptance Criteria (OAC)

To pass the production readiness audit, the following benchmarks must be met:

1.  **Model Loading**: The PyTorch model must load from the registry in **< 30 seconds**.
2.  **Inference Throughput**: The system must maintain **> 10 samples/sec** under batch load.
3.  **Observability**: The Grafana "Total Inspections" panel must show live data within 15 seconds of a scan.
4.  **Reproducibility**: A model version must be traceable to a specific **Git Commit Hash** and **DVC Data Version** in the MLflow UI.

## 5. Automated Execution
The test suite is automated via the CLI and integrated into the deployment pipeline:
```bash
python -m pytest src/tests/