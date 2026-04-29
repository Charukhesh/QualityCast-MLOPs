# 🛡️ QualityCast AI: Enterprise User Manual & Technical Documentation
> **Version 1.2.0 | Industrial-Grade MLOps Framework**

QualityCast-MLOps is a production-hardened computer vision ecosystem designed for real-time defect detection in industrial casting processes. This document serves as the primary technical reference for system deployment, operational management, and architectural oversight.

---

## 📂 Repository Structure & Data Provenance

The repository is organized according to the **"Clean Architecture"** principle, separating orchestration, business logic, and infrastructure configuration.

```text
QUALITYCAST-MLOPS/
├── .dvc/               # Internal DVC metadata (tracked by Git)
├── dags/               # Apache Airflow Directed Acyclic Graphs
│   └── qualitycast_dag.py  # Central pipeline orchestrator
├── data/               # Project Data Store
│   ├── raw/            # Incoming sensor data (DVC-tracked)
│   ├── processed/      # Augmented/Normalized data (DVC-tracked)
│   └── mlruns/         # MLflow tracking backend & artifact store
├── monitoring/         # Observability Configuration
│   ├── alert_rules.yml # Prometheus threshold definitions
│   └── prometheus.yml  # Scraper targets and global configs
├── src/                # Core Application Logic
│   ├── app/            # FastAPI (Inference) & Streamlit (UI)
│   │   ├── health_api.py # Orchestration probes (/health, /ready)
│   │   └── main.py       # Streamlit Dashboard & Monitoring logic
│   ├── ingestion/      # Data Engineering & Augmentation scripts
│   ├── tests/          # Pytest suite (Unit & Integration)
│   └── training/       # PyTorch Model training & MLflow logging
├── dvc.yaml            # DVC Pipeline Stage definitions
├── docker-compose.yaml # Multi-container orchestration manifest
├── feedback_log.csv    # Human-in-the-loop drift audit log
└── requirements.txt    # Standardized Python dependency manifest
```

---

## 🏗️ System Architecture & Logic Flow

The system adheres to a microservices-oriented architecture to ensure high availability and loose coupling.

1.  **Orchestration Layer:** Airflow manages the life-cycle from raw ingestion to model registration.
2.  **Versioning Layer:** DVC manages binary data lineage, ensuring every model is linked to a specific dataset hash.
3.  **Serving Layer:** FastAPI provides high-concurrency REST endpoints with automated readiness probes.
4.  **Observability Layer:** A Prometheus-Grafana stack provides real-time telemetry on system health and defect rates.

---

## 🚀 Installation & Deployment

### 1. Environment Initialization
Ensure the "No-Cloud" local environment has Docker and Docker Compose installed.

```bash
# Clone the repository
git clone https://github.com/Charukhesh/QualityCast-MLOPs.git
cd QualityCast-MLOps

# Initialize DVC (Decouples data from Git history)
dvc pull

# Deploy the containerized infrastructure
docker-compose up -d --build
```

### 2. Service Access Points
*   **Operator Dashboard:** `http://localhost:8501` (Streamlit)
*   **Pipeline Console:** `http://localhost:8080` (Airflow)
*   **Experiment Registry:** `http://localhost:5000` (MLflow)
*   **Operational Metrics:** `http://localhost:3001` (Grafana)

---

## 🕹️ Operational User Manual

### A. Managing the AI Pipeline
1.  Navigate to the **Airflow UI**.
2.  Locate `qualitycast_data_pipeline` and toggle it to **On**.
3.  Click **Trigger DAG**.
4.  **Audit Stages:**
    *   `validate_raw_data`: Monitors for sufficient data volume before training.
    *   `augment_images`: Executes versioned preprocessing via DVC.
    *   `train_classification_model`: Logs metrics/artifacts to MLflow.
    *   `check_api_ready_status`: Automatically verifies the inference engine is online.

### B. Performing Visual Inspections
1.  Open the **Streamlit Dashboard**.
2.  **Single Scan:** Upload an image of the casting front. The AI will output a **PASS ✅** or **REJECT ❌** status with a confidence score.
3.  **Batch Scan:** Upload a `.zip` archive for automated shift-end auditing. Download the generated CSV report for records.
4.  **Feedback Loop:** If the AI misidentifies a part, the operator must use the **"Report Prediction Error"** radio button. This writes to `feedback_log.csv` for future retraining triggers.

### C. Monitoring & Alerting
The system monitors three critical Key Performance Indicators (KPIs):

| Metric | Business Target | Alert Trigger |
| :--- | :--- | :--- |
| **Inference Latency** | < 200ms | ⚠️ Warning at > 200ms |
| **Throughput** | > 10 samples/sec | 🚨 Critical at < 5 samples/sec |
| **Defect Rate** | < 5% | 🛑 Shutdown at > 20% |

---

## 🛠️ Troubleshooting & FAQ

**Q: Airflow DAG fails at `augment_images` with "DVC command not found".**
*   *Solution:* Ensure DVC is listed in `requirements.txt`. The container must rebuild to include the DVC binary in its PATH.

**Q: Streamlit shows "MLflow Error: Model Offline".**
*   *Solution:* The latest model version has not been assigned the `production` alias. Go to MLflow UI -> Models -> Select Version -> Assign Alias: `production`.

**Q: Why is the Readiness probe (TC-02) returning 503?**
*   *Solution:* This is expected behavior during the first 30 seconds of startup while the PyTorch weights are being loaded from the registry into memory.

---

## 🧪 Technical Validation (QA)
To verify the system against the formal Test Plan (v1.2.0):
```bash
python -m pytest src/tests/
```
The suite validates 7 critical checkpoints (TC-01 to TC-07), ensuring the integrity of the data schema, API liveness, and model registry connectivity.
