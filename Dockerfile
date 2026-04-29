FROM apache/airflow:2.7.1

# Switch to root to install system dependencies if needed
USER root
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean

# Switch back to airflow user
USER airflow

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt