"""
Patient State & Feature Engineering Module
===========================================
Maintains a sliding window of patient telemetry ticks
and computes trajectory-aware features for real-time inference.
"""

from collections import deque


class PatientState:
    """
    Maintains a time-windowed buffer of patient telemetry readings.

    Computes trajectory features (velocity, acceleration) that capture
    the temporal dynamics of kidney function deterioration — a key
    contribution of the streaming telemetry approach.
    """

    def __init__(self, window_size=5):
        """
        Args:
            window_size: Number of recent ticks to retain in the sliding window.
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, tick):
        """Adds a new telemetry tick to the sliding-window buffer."""
        self.history.append(tick)

    def get_features(self):
        """
        Extracts the full feature set from the current tick plus
        computed trajectory metadata.

        The tick (from stream.py) already carries all CSV columns for
        the patient. This method enriches it with velocity features
        computed over the sliding window.

        Returns:
            dict: All patient features plus trajectory metadata,
                  or None if insufficient history for velocity calculation.
        """
        if len(self.history) < 2:
            return None  # Need at least 2 data points for velocity

        current = self.history[-1]
        oldest = self.history[0]

        # ── Trajectory features (novel contribution) ──
        egfr_velocity = current['eGFR'] - oldest['eGFR']
        creat_velocity = current['Serum_Creatinine'] - oldest['Serum_Creatinine']

        # eGFR acceleration (rate of velocity change) if enough history
        egfr_accel = 0.0
        if len(self.history) >= 3:
            mid = self.history[len(self.history) // 2]
            vel_recent = current['eGFR'] - mid['eGFR']
            vel_older = mid['eGFR'] - oldest['eGFR']
            egfr_accel = vel_recent - vel_older

        # ── Full feature dictionary ──
        # Copy all fields from the current tick (includes all CSV columns)
        features = dict(current)

        # Add computed trajectory metadata (for display and optional model input)
        features['eGFR_velocity'] = round(egfr_velocity, 2)
        features['Creatinine_velocity'] = round(creat_velocity, 2)
        features['eGFR_acceleration'] = round(egfr_accel, 2)

        return features