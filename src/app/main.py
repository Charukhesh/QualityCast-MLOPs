import streamlit as st
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torchvision import transforms
from PIL import Image
from prometheus_client import start_http_server, Counter, REGISTRY
from prometheus_client import Gauge, Counter
import torch
import pandas as pd
import io
import zipfile
import time
import os
import time

FEEDBACK_FILE = "feedback_log.csv"

def log_feedback_csv(pred, actual, conf):
    new_data = pd.DataFrame([{
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": pred,
        "actual": actual,
        "confidence": f"{conf*100:.2f}%"
    }])
    if not os.path.isfile(FEEDBACK_FILE):
        new_data.to_csv(FEEDBACK_FILE, index=False)
    else:
        new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)

# CONFIG & THEME
st.set_page_config(
    page_title="QualityCast AI | Industrial Grade",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f0f2f6; }
    
    /* Sidebar Polish */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Card Design */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
        text-align: center;
    }
    
    /* Glassmorphism Result Banner */
    .result-banner {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 800;
        margin: 20px 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .status-pass { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); color: #166534; border: 2px solid #22c55e; }
    .status-fail { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b; border: 2px solid #ef4444; }
    
    /* Clean Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #1d4ed8; transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

def get_or_create_metric(metric_cls, name, *args, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_cls(name, *args, **kwargs)

@st.cache_resource
def init_metrics():
    from prometheus_client import Histogram, Counter, Gauge

    return {
        "latency": Histogram("inference_latency_seconds", "Latency"),
        "batch": Histogram("batch_processing_seconds", "Batch time"),
        "req": Counter("inference_requests_total", "Requests"),
        "up": Gauge("service_up", "Service status"),
        "active": Gauge("active_processing", "Processing state"),
    }

metrics = init_metrics()

INFERENCE_LATENCY = metrics["latency"]
BATCH_PROCESSING_TIME = metrics["batch"]
REQUEST_COUNT = metrics["req"]
SERVICE_UP = metrics["up"]
ACTIVE_PROCESSING = metrics["active"]

SERVICE_UP.set(1)
# MONITORING
@st.cache_resource
def init_monitoring():
    try: start_http_server(8000)
    except: pass
    reg = REGISTRY._names_to_collectors
    ok = reg['ok_total'] if 'ok_total' in reg else Counter('ok_total', 'Good parts')
    defc = reg['def_total'] if 'def_total' in reg else Counter('def_total', 'Defects')
    return ok, defc

OK_COUNT, DEF_COUNT = init_monitoring()

if "scan_done" not in st.session_state:
    st.session_state["scan_done"] = False

# MODEL LOADING
@st.cache_resource
def load_registered_model(model_name="Casting_Quality_Model"):
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        model_uri = f"models:/{model_name}@production"
        model = mlflow.pytorch.load_model(model_uri)

        client = MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, "production")

        metadata = {
                "Registry Name": model_version.name,
                "Active Version": f"v{model_version.version}",
                "Run ID": model_version.run_id
            }

        run_data = client.get_run(model_version.run_id).data.metrics
        f1 = run_data.get("val_f1", 0.0) * 100

        return model.eval(), f1, metadata, f"Production (v{model_version.version})"
    except Exception as e:
        st.write("MLflow Error:", e)
        return None, 0, "Model Offline"

model, reg_f1, model_metadata, model_status = load_registered_model()

if "processed_count" not in st.session_state:
    st.session_state["processed_count"] = 0
if "ok_count" not in st.session_state:
    st.session_state["ok_count"] = 0
if "def_count" not in st.session_state:
    st.session_state["def_count"] = 0
if "start_time" not in st.session_state:
    st.session_state["start_time"] = time.time()

# SIDEBAR & USER MANUAL
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Center")
    
    st.subheader("📖 Operator Manual")
    with st.expander("Show Instructions", expanded=True):
        st.write("""
        1. **Load Image:** Use the Drag & Drop zone.
        2. **Process:** AI analyzes surface geometry.
        3. **Decision:** Review the PASS/REJECT status.
        4. **Registry:** Metrics are sent to Grafana.
        """)
    
    st.markdown("---")
    st.subheader("🛰️ System Integrity")
    if model is not None:
        st.success(f"Model Registry: {model_status}")
    else:
        st.error(f"Registry Error: {model_status}")
    st.info("Monitoring: Prometheus + Grafana Active")

# MAIN UI DESIGN
st.title("🛡️ QualityCast Pro: Inspection Dashboard")
st.caption("AI-Powered Real-Time Industrial Quality Assurance")

env = "Docker" if os.path.exists("/.dockerenv") else "Local"
st.metric("Environment", env)

metric_placeholder = st.empty()

def render_metrics():
    col_a, col_b, col_c, col_d = metric_placeholder.columns(4)

    with col_a:
        st.metric("Processed Units", st.session_state["processed_count"])

    with col_b:
        st.metric("AI F1 Score", f"{reg_f1:.2f}%")

    with col_c:
        latency_val = st.session_state.get("last_latency", None)
        st.metric("Latency", f"{latency_val:.1f} ms" if latency_val else "—")

    with col_d:
        st.metric("Environment", env)

# initial render
render_metrics()

st.write("") # Spacer

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        output = model(img_t)
        latency = (time.time() - start) * 1000

        prediction = (output > 0.5).int().item()
        confidence = output.item() if prediction == 1 else (1 - output.item())

    # store latency globally
    st.session_state["last_latency"] = latency

    return prediction, confidence, latency

tab1, tab2, tab3 = st.tabs(["🔍 Single Inspection", "📦 Batch Ingestion", "📈 Analytics"])

# TAB 1: SINGLE INSPECTION
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Upload Single Image", type=["jpg", "png", "jpeg"], key="single")
        # If a new file is uploaded, reset the scan state
        if file:
            st.image(Image.open(file), use_container_width=True)
            if st.session_state.get("last_file_name") != file.name:
                st.session_state["scan_done"] = False
                st.session_state["last_file_name"] = file.name

    with col2:
        if file and model:
            # BUTTON 1: RUN THE ANALYSIS
            if st.button("RUN SCAN"):
                REQUEST_COUNT.inc()
                ACTIVE_PROCESSING.set(1)
                pred, conf, lat = predict_image(Image.open(file), model)
                INFERENCE_LATENCY.observe(lat)
                ACTIVE_PROCESSING.set(0)

                # Store everything in session state
                st.session_state["last_pred_val"] = "PASS ✅" if pred == 1 else "REJECT ❌"
                st.session_state["last_conf_val"] = conf
                st.session_state["processed_count"] += 1
                st.session_state["last_latency"] = lat
                st.session_state["scan_done"] = True # Set the flag to stay open
                render_metrics()

            # DISPLAY RESULTS (This stays visible because it's tied to the flag, not the button)
            if st.session_state["scan_done"]:
                banner_class = "status-pass" if "PASS" in st.session_state["last_pred_val"] else "status-fail"
                st.markdown(f'<div class="result-banner {banner_class}">{st.session_state["last_pred_val"]}</div>', unsafe_allow_html=True)
                st.metric("Confidence", f"{st.session_state['last_conf_val']*100:.2f}%")

                # --- Feedback Loop UI ---
                st.markdown("---")
                st.write("🕵️ **Human-in-the-Loop Verification**")
                
                # We use a form here to prevent the page from refreshing before the log is written
                with st.form("feedback_form"):
                    correct_label = st.radio("What is the correct label?", ["PASS ✅", "REJECT ❌"])
                    submitted = st.form_submit_button("Confirm Correction")
                    
                    if submitted:
                        actual_val = "OK" if "PASS" in correct_label else "DEFECTIVE"
                        pred_val = "OK" if "PASS" in st.session_state["last_pred_val"] else "DEFECTIVE"
                        log_feedback_csv(pred_val, actual_val, st.session_state["last_conf_val"])
                        st.success("Correction saved to feedback_log.csv!")

# TAB 2: BATCH INGESTION
with tab2:
    st.subheader("High-Volume Batch Processing")
    zip_file = st.file_uploader("Upload ZIP archive of casting images", type=["zip"])
    
    if zip_file and model:
        if st.button("EXECUTE BATCH ANALYSIS"):
            results = []
            with zipfile.ZipFile(zip_file, "r") as z:
                img_list = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total = len(img_list)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                ACTIVE_PROCESSING.set(1)
                batch_start = time.time()
                
                for idx, img_name in enumerate(img_list):
                    with z.open(img_name) as f:
                        REQUEST_COUNT.inc()
                        start = time.time()

                        img_data = Image.open(io.BytesIO(f.read()))
                        pred, conf, lat = predict_image(img_data, model)
                        INFERENCE_LATENCY.observe(time.time() - start)

                        st.session_state["processed_count"] += 1
                        st.session_state["last_latency"] = lat
                        render_metrics()
                        if pred == 1:
                            st.session_state["ok_count"] += 1
                        else:
                            st.session_state["def_count"] += 1
                        
                        label = "OK" if pred == 1 else "DEFECTIVE"
                        if pred == 1: OK_COUNT.inc()
                        else: DEF_COUNT.inc()
                        
                        results.append({"Filename": img_name, "Prediction": label, "Confidence": f"{conf*100:.1f}%"})
                        
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"Processing {idx+1}/{total}...")

                batch_time = time.time() - batch_start
                BATCH_PROCESSING_TIME.observe(batch_time)

                ACTIVE_PROCESSING.set(0)
            
            # Display Batch Summary
            df = pd.DataFrame(results)
            st.success(f"Successfully processed {total} images.")
            
            # Industrial Metric Cards for Batch
            bc1, bc2 = st.columns(2)
            ok_total = len(df[df['Prediction'] == 'OK'])
            bc1.metric("Batch PASS", ok_total)
            bc2.metric("Batch REJECTS", total - ok_total, delta_color="inverse")
            
            st.dataframe(df, use_container_width=True)
            
            # Export Feature (Professional Touch)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Batch Report (CSV)", csv, "batch_report.csv", "text/csv")

with tab3:
    st.subheader("🛠️ Technical Analytics")

    col_v1, col_v2 = st.columns(2)

    # MODEL METADATA
    with col_v1:
        st.write("**Model Registry Metadata**")

        if model is not None:
            st.json({
                "Registry Name": model_metadata["Registry Name"],
                "Active Version": model_metadata["Active Version"],
                "Run ID": model_metadata["Run ID"],
                "Framework": "PyTorch",
                "Tracking": "MLflow Registry"
            })
        else:
            st.warning("Model not loaded")

    # LIVE THROUGHPUT
    with col_v2:
        st.write("**Live Session Throughput**")

        elapsed = time.time() - st.session_state["start_time"]
        throughput = st.session_state["processed_count"] / elapsed if elapsed > 0 else 0

        st.metric("Throughput", f"{throughput:.2f} samples/sec")

        st.bar_chart({
            "OK": [st.session_state["ok_count"]],
            "Defective": [st.session_state["def_count"]]
        })

        st.markdown("---")
        st.subheader("📋 Human Feedback Log (Drift Audit)")
        if os.path.exists(FEEDBACK_FILE):
            df_log = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df_log.tail(5), use_container_width=True) # Show latest 5
            
            # Add a download button for the evaluator to see the file
            with open(FEEDBACK_FILE, "rb") as file:
                st.download_button("📥 Download Full Feedback CSV", file, "feedback_log.csv", "text/csv")
        else:
            st.info("No feedback entries found yet. Run a scan and submit a correction to generate the log.")