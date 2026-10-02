"""Camera, sensor fusion and the live control loop."""
from contextlib import nullcontext
import time

import cv2

from .arduino import SensorReader
from .config import CONFIG_PATH, config_servo_limits
from .control import BalanceController, PIDPanel
from .sensors import PhotoEstimate, calibration_error, estimate_photo, fuse_positions
from .vision import BallTracker, position_pixel


def draw_measurements(frame, fused, camera, photo, highlight='fused'):
    """Three position values, with -- for an unavailable measurement."""
    cv2.rectangle(frame, (0, 0), (245, 91), (25, 25, 25), -1)
    for row, (source, label, value) in enumerate((
            ('fused', 'Fused', fused), ('camera', 'Camera', camera),
            ('photoresistor', 'Photoresistor', photo))):
        text = f'{label}: {value:+.1f} mm' if value is not None else f'{label}: --'
        colour = (100, 230, 110) if source == highlight else (240, 240, 240)
        cv2.putText(frame, text, (10, 24 + row*28), cv2.FONT_HERSHEY_SIMPLEX,
                    .53, colour, 1, cv2.LINE_AA)


def run(cap, link, cfg, config_path=CONFIG_PATH, preview=None):
    cfg = dict(cfg)  # Save is the only action that persists live edits.
    if preview not in (None, 'fused', 'camera', 'photoresistor'):
        raise ValueError('Preview must be camera, photoresistor, or fused')
    controlling = preview is None
    if not cfg.get('geometry'):
        raise ValueError('Run --calibrate camera first')
    if (controlling or preview in ('fused', 'photoresistor')) and link is None:
        raise ValueError('This mode requires --port')
    error = calibration_error(cfg)
    if error and link:
        print('Photoresistor unavailable:', error)
    if preview == 'photoresistor' and error:
        raise ValueError('Run --calibrate photoresistor first')
    fuse_positions(None, PhotoEstimate(), cfg)  # Validate before enabling hardware.
    if controlling and cfg.get('servo_calibrated') is not True:
        raise ValueError('Run --calibrate motor first')
    controller = BalanceController(cfg) if controlling else None
    tracker = BallTracker(cfg)
    previous_source = 'none'
    name = 'Beam balancer' if controlling else f'Preview: {preview}'
    panel = None
    try:
        if controlling:
            link.configure_limits(*config_servo_limits(cfg))
            panel = PIDPanel(cfg, controller, config_path)
            panel.open()
        cv2.namedWindow(name)
        with SensorReader(link) if link else nullcontext(None) as reader:
            while True:
                ok, frame = cap.read()
                frame_at = time.monotonic()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames')
                detected = tracker.update(frame, frame_at)
                # Pair fresh ADC readings by receipt time within 60 ms.
                sample = reader.nearest(frame_at) if reader else None
                now = time.monotonic()
                camera_age = now - frame_at
                photo_age = now - sample['received_at'] if sample else float('inf')
                camera_x = detected[1] if detected and camera_age < .20 else None
                photo = (estimate_photo(sample['adc'], cfg) if sample and 0 <= photo_age < .12
                         else PhotoEstimate())
                fused = fuse_positions(camera_x, photo, cfg, camera_age, photo_age)
                if controlling:
                    if fused.source != previous_source:
                        controller.reset_velocity()
                    previous_source = fused.source
                    angle = controller.level() if panel.paused else controller.update(fused.position_mm, now)
                    link.angle(angle)

                display = frame.copy()
                if controlling:
                    target = tuple(int(round(v)) for v in position_pixel(cfg['setpoint_mm'], cfg))
                    cv2.drawMarker(display, target, (255, 0, 255), cv2.MARKER_CROSS, 28, 2)
                if detected:
                    cv2.circle(display, detected[2], detected[3], (0, 255, 0), 2)
                draw_measurements(display, fused.position_mm, camera_x, photo.position_mm, preview or 'fused')
                cv2.imshow(name, display)
                if panel:
                    panel.show(now, fused.position_mm)
                key = cv2.waitKey(1) & 0xFF
                if panel:
                    panel.handle_key(key)
                    if panel.paused:
                        link.angle(cfg['servo_neutral_deg'])
                    if panel.quit:
                        break
                if key in (ord('q'), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        try:
            if controlling:
                link.stop()
        finally:
            if panel:
                panel.close()
