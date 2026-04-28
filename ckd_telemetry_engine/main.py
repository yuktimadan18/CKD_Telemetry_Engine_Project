import time
from src.stream import real_patient_stream
from src.features import PatientState
from src.inference import RiskModel
from src.alert import evaluate_alert


def run_engine():
    print("=" * 55)
    print("   CKD TELEMETRY ENGINE — ONLINE")
    print("   Model: Random Forest Classifier")
    print("=" * 55)

    
    stream = real_patient_stream(
        csv_path="data/Testing_CKD_dataset.csv",
        patient_row_index=5,
        health_drift=-0.2
    )

    state = PatientState(window_size=5)
    model = RiskModel(model_path="models/random_forest_ckd.pkl")

    print("\nListening for patient telemetry (REAL DATA SOURCE)...\n")

    
    try:
        while True:
            
            tick = next(stream)

            
            state.update(tick)

            
            features = state.get_features()

            if features:
                
                prediction = model.predict_risk(features)
                alert_status = evaluate_alert(prediction, tick)

                print(f"[{tick['timestamp']:.0f}] "
                      f"eGFR: {tick['eGFR']:6.2f} | "
                      f"Velocity: {features['eGFR_velocity']:+6.2f} | "
                      f"Risk: {prediction['risk_score']:.3f} | "
                      f"Stage: {prediction['stage_label']:25s} | "
                      f"{alert_status}")
            else:
                print(f"[{tick['timestamp']:.0f}] Calibrating baseline memory...")

            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nEngine safely shut down. Goodbye.")


if __name__ == "__main__":
    run_engine()