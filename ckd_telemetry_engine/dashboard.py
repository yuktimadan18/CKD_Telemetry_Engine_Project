import streamlit as st
import pandas as pd
import time
from src.stream import real_patient_stream
from src.features import PatientState
from src.inference import RiskModel
from src.alert import evaluate_alert

# ── 1. Page Configuration ──
st.set_page_config(page_title="CKD Telemetry", layout="wide", page_icon="🏥")
st.title("🏥 Real-Time CKD Telemetry Engine")
st.caption("Powered by Random Forest Classifier · Multi-Stage CKD Prediction")

# ── 2. Sidebar Controls ──
st.sidebar.header("Engine Controls")
patient_index = st.sidebar.number_input(
    "Select Patient ID (Row Index)",
    min_value=0, max_value=500, value=5, step=1
)
run_engine = st.sidebar.toggle("Start Live Telemetry", value=False)
st.sidebar.markdown("Flip the switch to begin monitoring the high-frequency data stream.")

# ── 3. Memory Reset Logic ──
# If the patient selection changes, wipe the state and re-initialize
if ('current_patient_index' not in st.session_state or
        st.session_state.current_patient_index != patient_index):
    st.session_state.current_patient_index = patient_index
    if 'stream' in st.session_state:
        del st.session_state['stream']

# ── 4. Initialize State ──
if 'stream' not in st.session_state:
    st.session_state.stream = real_patient_stream(
        csv_path="data/Testing_CKD_dataset.csv",
        patient_row_index=patient_index,
        health_drift=-0.15
    )
    st.session_state.state = PatientState(window_size=5)
    st.session_state.model = RiskModel(model_path="models/random_forest_ckd.pkl")
    st.session_state.history = pd.DataFrame(
        columns=['Tick', 'eGFR', 'Risk', 'Stage']
    )
    st.session_state.tick_counter = 0

st.markdown(f"Monitoring Patient: **hospital_{patient_index}**")

# ── 5. Create UI Layout ──
col1, col2, col3, col4 = st.columns(4)
egfr_metric = col1.empty()
velocity_metric = col2.empty()
risk_metric = col3.empty()
stage_metric = col4.empty()

st.markdown("---")
status_alert = st.empty()

# Stage probability distribution
st.markdown("### CKD Stage Probability Distribution")
proba_placeholder = st.empty()

st.markdown("### Physiological Trajectory (Last 50 Ticks)")
chart_placeholder = st.empty()


# ── 6. The Live Loop ──
if run_engine:
    while True:
        tick = next(st.session_state.stream)
        st.session_state.tick_counter += 1

        # Update sliding-window memory
        st.session_state.state.update(tick)
        features = st.session_state.state.get_features()

        if features:
            # Core inference pipeline
            prediction = st.session_state.model.predict_risk(features)
            alert_status = evaluate_alert(prediction, tick)

            # ── Update Big Number Metrics ──
            egfr_metric.metric("Current eGFR", f"{tick['eGFR']:.2f}")
            velocity_metric.metric(
                "Trajectory Velocity", f"{features['eGFR_velocity']:+.2f}"
            )
            risk_metric.metric("Failure Risk", f"{prediction['risk_score']:.1%}")
            stage_metric.metric("Predicted Stage", prediction['stage_label'])

            # ── Update Clinical Alert ──
            if "CRITICAL" in alert_status:
                status_alert.error(f"🚨 **{alert_status}**")
            elif "WARNING" in alert_status:
                status_alert.warning(f"⚠️ **{alert_status}**")
            else:
                status_alert.success(f"✅ **{alert_status}**")

            # ── Stage Probability Bar Chart ──
            if prediction.get('probabilities'):
                proba_df = pd.DataFrame({
                    'Stage': list(prediction['probabilities'].keys()),
                    'Probability': list(prediction['probabilities'].values())
                })
                proba_placeholder.bar_chart(proba_df.set_index('Stage'))

            # ── Update Live Trajectory Chart ──
            new_data = pd.DataFrame({
                'Tick': [st.session_state.tick_counter],
                'eGFR': [tick['eGFR']],
                'Risk': [prediction['risk_score']],
                'Stage': [prediction['stage']]
            })

            st.session_state.history = pd.concat(
                [st.session_state.history, new_data], ignore_index=True
            )

            # Keep only last 50 ticks for clean visualization
            if len(st.session_state.history) > 50:
                st.session_state.history = st.session_state.history.iloc[-50:]

            chart_data = st.session_state.history.set_index('Tick')
            chart_placeholder.line_chart(chart_data[['eGFR']])

        else:
            status_alert.info("⏳ Calibrating baseline physical memory...")

        # Throttle the loop
        time.sleep(0.5)