"""Phototransistor position, camera fusion and guided sensor calibration."""
from dataclasses import dataclass
import math
import statistics
import textwrap
import time
from typing import Optional

import cv2
import numpy as np

from .arduino import SensorReader
from .config import ball_travel_bounds, save_config
from .vision import BallTracker, axis_geometry, position_pixel


# Phototransistor position and calibration

@dataclass(frozen=True)
class PhotoEstimate:
    position_mm: Optional[float] = None
    confidence: float = 0.0
    coverage: tuple = ()
    reason: str = "no serial data"


def _vector(values, adc=False):
    return (isinstance(values, (list, tuple)) and len(values) == 8
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and (not adc or 0 <= v <= 1023)
                    for v in values))


def calibration_error(cfg):
    positions = cfg.get("sensor_positions_mm")
    baseline = cfg.get("sensor_baseline")
    covered = cfg.get("sensor_full_cover")
    if not _vector(positions) or len(set(positions)) != 8:
        return "set eight distinct sensor positions in channel order"
    if not _vector(baseline, adc=True) or not _vector(covered, adc=True):
        return "run --calibrate photoresistor (uncovered + ball-covered)"
    polarity = cfg.get("sensor_shadow_polarity", 1)
    if polarity not in (-1, 1):
        return "sensor_shadow_polarity must be +1 or -1"
    if any((full - base) * polarity < 40 for base, full in zip(baseline, covered)):
        return "each sensor needs at least 40 ADC counts of calibrated response"
    noise = cfg.get("sensor_noise_adc", [0] * 8)
    if not _vector(noise) or any(n < 0 for n in noise):
        return "invalid sensor noise calibration"
    return None


def estimate_photo(readings, cfg):
    """Coverage is response fraction; confidence is a relative blend weight.

    Neither represents a probability or a measured ball-centre accuracy.
    """
    error = calibration_error(cfg)
    if error:
        return PhotoEstimate(reason=error)
    if not _vector(readings, adc=True):
        return PhotoEstimate(reason="invalid/no serial data")
    deadband = cfg.get("photo_deadband", 0.08)
    min_peak = cfg.get("photo_min_peak", 0.20)
    power = cfg.get("photo_weight_power", 1.0)
    if not (0 <= deadband < min_peak <= 1 and 0.25 <= power <= 4):
        raise ValueError("Require 0 <= photo_deadband < photo_min_peak <= 1; photo_weight_power 0.25..4")
    baseline, covered = cfg["sensor_baseline"], cfg["sensor_full_cover"]
    spans = [full - base for base, full in zip(baseline, covered)]
    raw = [(reading - base) / span for reading, base, span in zip(readings, baseline, spans)]
    coverage = tuple(max(0.0, min(1.0, value)) for value in raw)

    def reject(reason):
        return PhotoEstimate(coverage=coverage, reason=reason)

    if min(raw) < -0.25 or max(raw) > 1.35:
        return reject("outside calibrated light range")
    if max(coverage) < min_peak:
        return reject("weak/no shadow")
    noise = cfg.get("sensor_noise_adc", [0] * 8)
    floors = [max(deadband, 3 * n / abs(span)) for n, span in zip(noise, spans)]
    weights = [max(0.0, (c - floor) / (1 - floor)) ** power if floor < 1 else 0.0
               for c, floor in zip(coverage, floors)]
    positions = cfg["sensor_positions_mm"]
    order = sorted(range(8), key=positions.__getitem__)
    active = [rank for rank, i in enumerate(order) if weights[i] > 0]
    if not active:
        return reject("below noise floor")
    # A faint response elsewhere on the board must not invalidate a strong
    # local ball shadow. Group by physical adjacency, then require exactly one
    # group to reach the same minimum peak used to recognise a ball at all.
    clusters = []
    for rank in active:
        if not clusters or rank != clusters[-1][-1] + 1:
            clusters.append([])
        clusters[-1].append(rank)
    strong = [cluster for cluster in clusters
              if max(coverage[order[rank]] for rank in cluster) >= min_peak]
    if not strong:
        return reject("below noise floor")
    if len(strong) > 1:
        return reject("separated shadows")
    active = strong[0]
    if len(active) > 3:
        return reject("broad shadow / lighting change")
    selected = [order[rank] for rank in active]
    total = sum(weights[i] for i in selected)
    position = sum(positions[i] * weights[i] for i in selected) / total
    confidence = max(coverage[i] for i in selected)
    # Saturation covers a range of possible centres. Do not snap to a sensor.
    if sum(coverage[i] >= 0.95 for i in selected) > 1:
        confidence *= 0.6
    if active == [0] or active == [7]:
        confidence *= 0.5  # Cannot resolve travel beyond an array endpoint.
    reason = "ok; weak separate shadow ignored" if len(clusters) > 1 else "ok"
    return PhotoEstimate(position, confidence, coverage, reason)


def summarize_levels(samples):
    if len(samples) < 20:
        raise RuntimeError("Not enough fresh sensor data; check serial connection and retry")
    columns = list(zip(*samples))
    medians = [statistics.median(column) for column in columns]
    noise = [1.4826 * statistics.median(abs(v - centre) for v in column)
             for column, centre in zip(columns, medians)]
    return [round(v) for v in medians], noise


def calibrate_sensors(link, cfg, config_path, cap):
    """Record baseline and covered levels using the live camera guide."""
    with SensorCalibrationView(cap, link, cfg) as view:
        return _calibrate_sensors(cfg, config_path, view)


def _calibrate_sensors(cfg, config_path, view):
    positions = cfg.get("sensor_positions_mm")
    if not _vector(positions) or len(set(positions)) != 8:
        raise ValueError("Set the eight distinct sensor_positions_mm in channel order first")
    polarity = cfg.get("sensor_shadow_polarity", 1)
    if polarity not in (-1, 1):
        raise ValueError("sensor_shadow_polarity must be +1 or -1")
    print("Support the beam level. This mode does not drive the servo.")
    print("Use the actual ball for coverage; keep fingers and other shadows off the PCB.")
    print("Click the camera window. Space / Enter records 1.5 seconds; Q / Esc cancels without saving.")

    def notice(message):
        print(message)
        view.notice = message

    result = view.capture(None)
    if result is None:
        return False
    baseline, baseline_noise = result
    full_cover, noise = [], baseline_noise[:]
    for i in range(len(positions)):
        while True:
            result = view.capture(i)
            if result is None:
                print("Cancelled; previous calibration kept.")
                return False
            levels, level_noise = result
            span = (levels[i] - baseline[i]) * polarity
            measured_noise = max(baseline_noise[i], level_noise[i])
            if span < max(40, 6 * measured_noise):
                notice(f"Y{i}: response {span:g} ADC too small/noisy. Check ball position, wiring, polarity; retry.")
                continue
            if measured_noise > max(10, span * 0.10):
                notice(f"Y{i}: readings unstable; hold the ball still and retry.")
                continue
            full_cover.append(levels[i])
            noise[i] = measured_noise
            notice(f"Y{i} captured")
            break
    # Both levels are saved together, only after every channel succeeds.
    updated = dict(cfg, sensor_baseline=baseline, sensor_full_cover=full_cover,
                   sensor_noise_adc=[round(n, 2) for n in noise])
    save_config(config_path, updated)
    cfg.update(updated)
    print("Photoresistor calibration saved.")
    return True


# Camera and phototransistor fusion

@dataclass(frozen=True)
class FusedPosition:
    position_mm: Optional[float] = None
    source: str = "none"
    camera_weight: float = 0.0
    photo_weight: float = 0.0
    disagreement_mm: Optional[float] = None
    reason: str = "no fresh measurements"


def fuse_positions(camera_mm, photo, cfg, camera_age=0.0, photo_age=0.0):
    """Use camera agreement to discount photo, without adapting its calibration.

    Ages are host receipt ages, not exposure timestamps. A bad camera estimate
    can still mislead this rule; preview should establish the camera reference.
    Returned weights are normalized contributions, not accuracy probabilities.
    """
    camera_gain = cfg.get("fusion_camera_weight", 1.0)
    photo_gain = cfg.get("fusion_photo_weight", 1.0)
    scale = cfg.get("fusion_agreement_mm", 6.0)
    reject_at = cfg.get("fusion_reject_mm", 24.0)
    if not all(math.isfinite(v) for v in (camera_gain, photo_gain, scale, reject_at)):
        raise ValueError("Fusion settings must be finite")
    if min(camera_gain, photo_gain) < 0 or max(camera_gain, photo_gain) <= 0 or not 0 < scale < reject_at:
        raise ValueError("Fusion weights must be nonnegative (at least one positive); require 0 < agreement < reject mm")

    def freshness(age, limit):
        if not math.isfinite(age) or not 0 <= age < limit:
            return 0.0
        return 1.0 - age / limit

    camera_ok = camera_mm is not None and math.isfinite(camera_mm)
    photo_ok = (photo.position_mm is not None and math.isfinite(photo.position_mm)
                and math.isfinite(photo.confidence) and photo.confidence > 0)
    cw = camera_gain * freshness(camera_age, 0.20) if camera_ok else 0.0
    pw = photo_gain * min(1.0, photo.confidence) * freshness(photo_age, 0.12) if photo_ok else 0.0
    delta = photo.position_mm - camera_mm if cw > 0 and pw > 0 else None
    reason = "ok"
    if delta is not None:
        if abs(delta) >= reject_at:
            pw = 0.0
            reason = "photo/camera disagreement: camera used"
        else:
            pw *= 1.0 / (1.0 + (delta / scale) ** 2)
    if cw + pw <= 0:
        return FusedPosition()
    cw, pw = cw / (cw + pw), pw / (cw + pw)
    position = (cw * camera_mm if cw else 0.0) + (pw * photo.position_mm if pw else 0.0)
    source = "fused" if cw and pw else "camera" if cw else "photo"
    return FusedPosition(position, source, cw, pw, delta, reason)


# Sensor calibration camera view

def draw_sensor_axis(display, cfg, active_channel=None):
    pivot, scale = axis_geometry(cfg)
    unit = scale / np.linalg.norm(scale)
    normal = np.array([-unit[1], unit[0]])
    positions = cfg['sensor_positions_mm']

    def pixel(point):
        return tuple(int(round(v)) for v in point)

    cv2.arrowedLine(display, pixel(pivot + scale * (min(positions)-5)),
                    pixel(pivot + scale * (max(positions)+5)), (255, 220, 0), 2,
                    tipLength=0.025)
    for channel, mm in enumerate(positions):
        point = pivot + scale * mm
        active = channel == active_channel
        colour = (0, 230, 255) if active else (255, 220, 0)
        size = 42 if active else 7
        cv2.line(display, pixel(point - normal*size), pixel(point + normal*size),
                 colour, 2 if active else 1)
        cv2.circle(display, pixel(point), 7 if active else 3, colour, 2)
        label_point = point + normal*20
        label = f'Y{channel}'
        cv2.putText(display, label, pixel(label_point + [-10, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, colour, 1, cv2.LINE_AA)
        cv2.putText(display, f'{mm:+g}', pixel(label_point + [-14, 17]),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, colour, 1, cv2.LINE_AA)
    cv2.drawMarker(display, pixel(pivot), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(display, '0 mm / pivot', pixel(pivot + normal*57 + [-35, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)


class SensorCalibrationView:
    def __init__(self, cap, link, cfg):
        axis_geometry(cfg)
        self.travel_bounds = ball_travel_bounds(cfg)
        self.cap, self.cfg = cap, cfg
        self.reader = SensorReader(link)
        self.tracker = BallTracker(cfg)
        self.name = 'Photoresistor calibration'
        self.notice = 'Keep the beam level and lighting steady.'

    def __enter__(self):
        cv2.namedWindow(self.name)
        self.reader.__enter__()
        return self

    def __exit__(self, *args):
        self.reader.__exit__(*args)
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass

    def capture(self, channel):
        """Keep rendering during positioning and capture; Q cancels either."""
        started = None
        duration = 1.5
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('Camera stopped during sensor calibration; previous calibration kept')
            detected = self.tracker.update(frame, time.monotonic())
            now = time.monotonic()
            sample = self.reader.nearest(now, max_age=.12)
            display = frame.copy()
            if detected:
                cv2.circle(display, detected[2], detected[3], (0, 255, 0), 2)
            draw_sensor_axis(display, self.cfg, channel)
            target = None if channel is None else self.cfg['sensor_positions_mm'][channel]
            prompt = ('Step 1/9: Remove ball - uncovered baseline' if channel is None else
                      f'Step {channel+2}/9: Centre ball at Y{channel} / Q{channel+1}, {target:+g} mm')
            if target is not None:
                reachable = max(self.travel_bounds[0], min(self.travel_bounds[1], target))
                if reachable != target:
                    # The outer sensor can be covered without placing the ball
                    # centre beyond the physical end stop.
                    prompt = f'Step {channel+2}/9: Cover Y{channel} / Q{channel+1}; ball centre {reachable:+g} mm'
                    marker = tuple(int(round(v)) for v in position_pixel(reachable, self.cfg))
                    cv2.drawMarker(display, marker, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            status = ('SPACE / ENTER: record  |  Q / ESC: cancel' if started is None else
                      f'RECORDING: hold still  {max(0, duration-(now-started)):.1f}s left  |  Q: cancel')
            camera_text = f'Camera: {detected[1]:+.1f} mm' if detected else 'Camera: --'
            # Keep the instructions clear against a bright background.
            cv2.rectangle(display, (0, 0), (display.shape[1], 86), (25, 25, 25), -1)
            for row, text in enumerate((prompt, status, camera_text)):
                cv2.putText(display, text, (10, 23 + row*26), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (255, 255, 255), 1, cv2.LINE_AA)
            bottom_lines = textwrap.wrap(self.notice, width=85)
            y = display.shape[0] - len(bottom_lines)*20 - 8
            cv2.rectangle(display, (0, y-17), (display.shape[1], display.shape[0]), (25, 25, 25), -1)
            for row, text in enumerate(bottom_lines):
                cv2.putText(display, text, (10, y+row*20), cv2.FONT_HERSHEY_SIMPLEX,
                            .40, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.imshow(self.name, display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if started is None:
                if key in (32, 10, 13):
                    if sample is None:
                        self.notice = 'No fresh ADC data: check serial connection, then retry.'
                    else:
                        started = time.monotonic()
                        self.notice = 'Recording ADC; keep the ball and lighting steady.'
            elif now - started >= duration:
                samples = self.reader.between(started, started + duration)
                # A camera stall must not silently select a stale partial burst.
                if (len(samples) < 20 or samples[-1]['received_at']-samples[0]['received_at'] < 1.0
                        or max((b['received_at']-a['received_at'] for a, b in zip(samples, samples[1:])), default=0) > .2):
                    started = None
                    self.notice = 'Not enough continuous fresh ADC data. Check connection; Space retries.'
                    continue
                return summarize_levels([sample['adc'] for sample in samples])
