"""
Clinical Alert Module
=====================
Translates ML-predicted CKD stage and risk scores
into actionable clinical alerts for hospital staff.
"""


def evaluate_alert(prediction, tick_data):
    """
    Translates the model's prediction into a human-readable clinical alert.

    Args:
        prediction: dict from RiskModel.predict_risk() with keys:
                    'risk_score', 'stage', 'stage_label', 'probabilities'
                    — OR a float (legacy backward compatibility)
        tick_data:  raw telemetry tick with eGFR value

    Returns:
        str: Clinical alert string with severity level
    """
    # Handle structured prediction dict
    if isinstance(prediction, dict):
        stage = prediction.get('stage', 0)
        label = prediction.get('stage_label', 'Unknown')
        risk = prediction.get('risk_score', 0)
    else:
        # Backward compatibility: bare float risk score
        risk = float(prediction)
        stage = (0 if risk < 0.25 else
                 1 if risk < 0.45 else
                 2 if risk < 0.65 else
                 3 if risk < 0.85 else 4)
        label = f"Stage {stage}"

    egfr = tick_data.get('eGFR', 999)

    # Stage-based clinical alerting
    if stage >= 4:
        return (f"CRITICAL: {label} — Risk {risk:.1%} — "
                f"eGFR {egfr:.1f} — Immediate Dialysis/Transplant Evaluation!")
    elif stage >= 3:
        return (f"CRITICAL: {label} — Risk {risk:.1%} — "
                f"eGFR {egfr:.1f} — Urgent Nephrologist Referral!")
    elif stage >= 2:
        return (f"WARNING: {label} — Risk {risk:.1%} — "
                f"eGFR {egfr:.1f} — Close Monitoring Required")
    elif stage >= 1:
        return (f"WARNING: {label} — Risk {risk:.1%} — "
                f"eGFR {egfr:.1f} — Lifestyle Modification Recommended")
    else:
        return f"STABLE: {label} — Risk {risk:.1%} — eGFR {egfr:.1f}"