from __future__ import annotations

import json
import time


def replay_stream(model, df, delay_seconds: float = 0.0):
    """Replay transactions one by one and yield alert dictionaries."""
    for _, row in df.iterrows():
        one = row.to_frame().T
        alerts = model.predict_alerts(one)
        for alert in alerts:
            yield alert
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def print_json_alerts(model, df, delay_seconds: float = 0.0):
    for alert in replay_stream(model, df, delay_seconds):
        print(json.dumps(alert, default=str))
