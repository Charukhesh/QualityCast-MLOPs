from fastapi import FastAPI, Response, status
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
import time

app = FastAPI()

# ORCHESTRATION METRICS 
# 1 = Healthy/Ready, 0 = Unhealthy/Loading
LIVENESS = Gauge("model_service_alive", "Process is running")
READINESS = Gauge("model_service_ready", "Model is loaded and ready for inference")

# Global variable to track model state
MODEL_STATE = {"loaded": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on STARTUP
    try:
        # load_model_logic_here()
        MODEL_STATE["loaded"] = True
        READINESS.set(1)
        LIVENESS.set(1)
    except Exception:
        MODEL_STATE["loaded"] = False
        LIVENESS.set(1)
    
    yield 
    pass

app = FastAPI(lifespan=lifespan)

@app.get("/health", status_code=200)
def health():
    """Liveness Probe: Is the web server responding?"""
    LIVENESS.set(1)
    return {"status": "healthy"}

@app.get("/ready")
def ready(response: Response):
    """Readiness Probe: Is the model loaded and ready for traffic?"""
    if MODEL_STATE["loaded"]:
        return {"status": "ready"}
    else:
        # 503 Service Unavailable tells orchestrators NOT to send traffic here yet
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "detail": "Model weights are still loading"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)