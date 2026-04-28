import numpy as np
import time
import pandas as pd

def real_patient_stream(csv_path="data/Testing_CKD_dataset.csv", patient_row_index=0, health_drift=-0.1):
    """Loads a real patient from the clinical dataset and simulates a live telemetry stream."""
    
    # 1. Read the real data
    try:
        df = pd.read_csv(csv_path)
        # Check if the row index exists
        if patient_row_index >= len(df):
            patient_row_index = 0
            
        base_patient = df.iloc[patient_row_index].to_dict()
    except FileNotFoundError:
        # Fallback if file isn't there
        print(f"CRITICAL: Could not find {csv_path}. Using fallback values.")
        base_patient = {"Age": 50, "eGFR": 85, "Serum_Creatinine": 1.1}
        
    # 2. Extract starting points
    egfr = base_patient.get('eGFR', 85.0)
    creat = base_patient.get('Serum_Creatinine', 1.1)
    patient_id = f"PATIENT_{patient_row_index}"
    
    while True:
        # 3. Apply physics
        egfr += health_drift + np.random.normal(0, 0.5)
        creat_drift = abs(health_drift) * 0.02
        creat += creat_drift + np.random.normal(0, 0.05)
        
        if np.random.rand() < 0.01: 
            egfr -= np.random.uniform(2.0, 5.0)
            
        egfr = max(5.0, egfr)
        
        # 4. Return the full profile
        live_payload = base_patient.copy() 
        live_payload.update({
            "timestamp": time.time(),
            "patient_id": patient_id,
            "eGFR": round(egfr, 2),
            "Serum_Creatinine": round(creat, 2)
        })
        
        yield live_payload