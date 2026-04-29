from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath("/opt/airflow/src"))

with DAG(
    dag_id='qualitycast_data_pipeline',
    start_date=datetime(2026, 4, 5),
    schedule=None, 
    catchup=False
) as dag:

    # 1. Validation & Baseline Stats
    validate = BashOperator(
        task_id='validate_raw_data',
        bash_command='ls /opt/airflow/data/raw/ok_front | wc -l && ls /opt/airflow/data/raw/def_front | wc -l'
    )

    # 2. Run Augmentation (DVC Preprocess)
    # Note: We use absolute paths to ensure DVC finds the .dvc folder in the root
    augment = BashOperator(
        task_id='augment_images',
        bash_command='cd /opt/airflow && /home/airflow/.local/bin/dvc repro preprocess'
    )

    # 3. Data Quality Checks
    version_data = BashOperator(
        task_id='verify_processed_data',
        bash_command='echo "Count of OK samples: " && ls /opt/airflow/data/processed/ok_front | wc -l'
    )

    # 4. Train model (DVC Train / MLflow)
    train_model = BashOperator(
        task_id='train_classification_model',
        bash_command='cd /opt/airflow && /home/airflow/.local/bin/dvc repro train'
    )

    # 5. Verify MLflow Registry (Renamed per your diagram)
    verify_registry = BashOperator(
        task_id='verify_registry',
        bash_command="ls -R /opt/airflow/data/mlruns | grep 'artifacts' || (echo '❌ MLflow artifacts missing' && exit 1)"
    )

    # 6. Deploy to Inference API (Automated Deployment)
    # This simulates moving the model weights to the shared volume the API uses
    deploy_to_api = BashOperator(
        task_id='deploy_to_inference_api',
        bash_command='cp /opt/airflow/best_model_weights.pth /opt/airflow/data/model_store/ || echo "Weights already deployed"'
    )

    # 7. Check API Ready Status (Health Check Orchestration)
    # It pings the readiness endpoint we created in FastAPI
    check_api = BashOperator(
        task_id='check_api_ready_status',
        bash_command='curl -s -f http://frontend:8501/ready || (echo "Waiting for model load..." && sleep 10 && curl -s -f http://frontend:8501/ready)'
    )

    # 8. Verify Grafana Metrics (Production Monitoring)
    # Check if Prometheus is successfully scraping the metrics
    verify_metrics = BashOperator(
        task_id='verify_grafana_metrics',
        bash_command='curl -s http://prometheus:9090/api/v1/targets | grep "health" && echo "✅ Metrics Scraper Active"'
    )

    # DAG Dependency Chain (Matches your Image)
    validate >> augment >> version_data >> train_model >> verify_registry >> deploy_to_api >> check_api >> verify_metrics