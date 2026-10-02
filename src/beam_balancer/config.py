"""Load project settings and atomically save hardware calibration."""
import json
import math
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "beam_config.json"


def ball_travel_bounds(cfg):
    """Ball-centre travel between end stops, with the pivot at rail centre."""
    length, diameter = cfg.get('rail_length_mm'), cfg.get('ball_diameter_mm')
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in (length, diameter)):
        raise ValueError('Set finite rail_length_mm and ball_diameter_mm')
    if not 0 < diameter < length:
        raise ValueError('Require 0 < ball_diameter_mm < rail_length_mm')
    reach = (length - diameter) / 2
    return -reach, reach


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def save_config(path, cfg):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cfg, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_servo_limits(low, neutral, high):
    values = (low, neutral, high)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or int(v) != v for v in values):
        raise ValueError("Servo positions must be whole degrees")
    if not 0 <= low < neutral < high <= 180:
        raise ValueError("Require 0 <= minimum < level < maximum <= 180 degrees")
    return tuple(int(v) for v in values)


def config_servo_limits(cfg):
    return validate_servo_limits(cfg["servo_min_deg"], cfg["servo_neutral_deg"],
                                 cfg["servo_max_deg"])
