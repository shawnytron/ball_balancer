"""Camera coordinates, pale-ball tracking and camera calibration."""
import math
import time

import cv2
import numpy as np

from .config import save_config


# Beam coordinates

def axis_geometry(cfg):
    try:
        geometry = cfg['geometry']
        left, pivot, right = (np.asarray(geometry[key], dtype=float)
                              for key in ('left_px', 'pivot_px', 'right_px'))
        length = float(cfg['beam_length_mm'])
        if (any(p.shape != (2,) or not np.all(np.isfinite(p)) for p in (left, pivot, right))
                or not np.isfinite(length) or length <= 0 or np.linalg.norm(right-left) < 20):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('Run --calibrate camera first so the sensor axis can be placed on the image') from None
    return pivot, (right - left) / length


def position_pixel(position_mm, cfg):
    """Inverse of the camera's projected mm coordinate, on its pivot axis."""
    pivot, pixels_per_mm = axis_geometry(cfg)
    return pivot + pixels_per_mm * position_mm


# Ball detection and camera calibration

def geometry_position(point, cfg):
    geometry = cfg.get("geometry", {})
    if not all(k in geometry for k in ("left_px", "pivot_px", "right_px")):
        return None
    left = np.array(geometry["left_px"], dtype=float)
    pivot = np.array(geometry["pivot_px"], dtype=float)
    right = np.array(geometry["right_px"], dtype=float)
    axis = right - left
    span = float(np.linalg.norm(axis))
    if span < 20 or cfg["beam_length_mm"] <= 0:
        return None
    unit = axis / span
    return float(np.dot(np.array(point, dtype=float) - pivot, unit)
                 * cfg["beam_length_mm"] / span)


def _on_beam(x, y, cfg):
    geo = cfg.get("geometry", {})
    if not all(key in geo for key in ("left_px", "pivot_px", "right_px")):
        return False
    left = np.asarray(geo["left_px"], dtype=float)
    axis = np.asarray(geo["right_px"], dtype=float) - left
    span = float(np.linalg.norm(axis))
    if span < 20:
        return False
    offset = np.array([x, y]) - left
    distance = abs(axis[0] * offset[1] - axis[1] * offset[0]) / span
    along = float(np.dot(offset, axis) / span**2)
    reference_length = cfg["beam_length_mm"]
    if reference_length <= 0:
        return False
    # Scale references need not also be hard detection boundaries. Allow a
    # small physical margin without changing the pixel-to-millimetre scale.
    margin = max(0, cfg.get("beam_end_margin_mm", 0)) / reference_length
    return (distance <= cfg["beam_roi_half_width_px"]
            and -margin <= along <= 1 + margin)


def _outline_candidates(frame, colour_mask, cfg):
    """Use a visible circular boundary with colour support, not white alone.

    A pale centre or a colour region touching the background need not form
    one clean contour. Pale mode allows a small coloured rim around a white
    centre. Completely colourless objects and circles off the beam are rejected.
    """
    minimum, maximum = cfg["ball_radius_px"]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 1)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(10, minimum * 2),
        param1=80, param2=30, minRadius=max(1, int(minimum)),
        maxRadius=int(maximum),
    )
    if circles is None:
        return []
    candidates = []
    minimum_support = 0.02 if cfg.get("ball_pale_mode", False) else 0.15
    height, width = colour_mask.shape
    for x, y, radius in circles[0]:
        if not _on_beam(x, y, cfg):
            continue
        position = geometry_position((x, y), cfg)
        if position is None:
            continue
        # Clip to the frame, but reject truncated circles: their centres are
        # unreliable and the full beam should remain in view.
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            continue
        x0, x1 = int(x - radius), int(x + radius) + 1
        y0, y1 = int(y - radius), int(y + radius) + 1
        yy, xx = np.ogrid[y0:y1, x0:x1]
        # A nearly white ball can retain colour only at its very edge.
        # Cropping to 95% of the radius discarded precisely that evidence.
        disk = (xx - x)**2 + (yy - y)**2 <= radius**2
        support = float(np.mean(colour_mask[y0:y1, x0:x1][disk] > 0))
        if support > minimum_support:
            detection = (math.pi * float(radius)**2, position,
                         (round(float(x)), round(float(y))), round(float(radius)))
            candidates.append((support, detection))
    return candidates


def _blur_candidates(hsv, colour_mask, cfg, previous):
    """Recover a modestly blurred pale ball only during an established track.

    White highlights may fill the shape, but measured colour must still be
    present. Size, compactness, beam alignment and temporal association bound
    the fallback; a white circle alone is never sufficient.
    """
    bright = cv2.inRange(hsv, np.array([0, 0, 190], np.uint8),
                        np.array([179, 65, 255], np.uint8))
    shape = cv2.bitwise_or(colour_mask, bright)
    shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum, maximum = cfg['ball_radius_px']
    axis = np.subtract(cfg['geometry']['right_px'], cfg['geometry']['left_px'])
    beam_angle = math.atan2(axis[1], axis[0])
    height, width = shape.shape
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(contour) < 5 or area < 0.7 * math.pi * minimum**2:
            continue
        (x, y), (w, h), angle = cv2.fitEllipse(contour)
        minor, major = min(w, h)/2, max(w, h)/2
        if not minimum <= minor <= maximum or major > 1.6*minor:
            continue
        if abs(minor-previous[3]) > 0.25*previous[3] or not _on_beam(x, y, cfg):
            continue
        if not 0.8 <= area/(math.pi*minor*major) <= 1.1:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area/hull_area < 0.93:
            continue
        major_angle = math.radians(angle + (90 if h > w else 0))
        if major/minor > 1.15 and abs(math.cos(major_angle-beam_angle)) < 0.87:
            continue
        x0, y0, bw, bh = cv2.boundingRect(contour)
        if x0 <= 0 or y0 <= 0 or x0+bw >= width or y0+bh >= height:
            continue
        inside = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(inside, [contour - (x0, y0)], -1, 255, -1)
        coloured = (colour_mask[y0:y0+bh, x0:x0+bw] > 0) & (inside > 0)
        support = np.count_nonzero(coloured)/max(1, np.count_nonzero(inside))
        if support < 0.02:
            continue
        # Reject a white reflection with just one tiny coloured contact point.
        cy, cx = np.nonzero(coloured)
        sectors = ((np.arctan2(cy+y0-y, cx+x0-x)+math.pi)*4/math.pi).astype(int) % 8
        if np.count_nonzero(np.bincount(sectors, minlength=8) >= 3) < 3:
            continue
        position = geometry_position((x, y), cfg)
        if position is not None:
            candidates.append((support, (area, position, (round(x), round(y)), round(minor))))
    return candidates


def select_candidate(candidates, previous=None, max_distance_px=60):
    """Prefer continuity after acquisition; reject position/size outliers."""
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item[0])[1]
    choices = []
    for confidence, detection in candidates:
        distance = math.hypot(detection[2][0] - previous[2][0],
                              detection[2][1] - previous[2][1])
        size_change = abs(detection[3] - previous[3]) / max(1, previous[3])
        if distance > max_distance_px or size_change > 0.25:
            continue
        # Distance and size take priority over brightness/colour strength.
        cost = distance / max_distance_px + size_change
        choices.append((cost, -confidence, detection))
    return min(choices, key=lambda item: item[:2])[2] if choices else None


def detect_ball(frame, cfg, previous=None, max_distance_px=60):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["ball_hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["ball_hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if cfg.get("ball_pale_mode", False) and previous is not None:
        # Motion spreads a thin coloured rim over white pixels, reducing its
        # saturation. This relaxed mask is used only for a constrained shape
        # near an already observed ball, never for initial acquisition.
        faint_lower = lower.copy()
        faint_lower[1] = max(4, int(lower[1])//2)
        faint_mask = cv2.inRange(hsv, faint_lower, upper)
        selected = select_candidate(_blur_candidates(hsv, faint_mask, cfg, previous),
                                    previous, max_distance_px)
        if selected is not None:
            # Fit the complete bright body; a Hough circle can fit one end of
            # a blurred silhouette and shift the reported centre backward.
            return selected
    outlines = _outline_candidates(frame, mask, cfg)
    selected = select_candidate(outlines, previous, max_distance_px)
    if selected is not None:
        return selected
    if outlines or cfg.get("ball_pale_mode", False):
        # Pale colours match many background surfaces; require a circular
        # outline instead of accepting a loosely shaped colour blob.
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    minimum, maximum = cfg["ball_radius_px"]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not (minimum <= radius <= maximum) or area < 0.35 * math.pi * radius**2:
            continue
        position = geometry_position((x, y), cfg)
        if position is None:
            continue
        if not _on_beam(x, y, cfg):
            continue
        candidates.append((area, (area, position, (int(x), int(y)), int(radius))))
    return select_candidate(candidates, previous, max_distance_px)


def ball_colour_bounds(sample, previous_bounds=None):
    """Accept a pale centre while retaining hue evidence from the coloured rim."""
    hue, saturation, value = map(float, sample)
    if saturation < 40:
        # Hue is meaningless in a fully white pixel. Keep the earlier colour
        # range (or the project's orange/yellow default) in that case.
        previous_lower, previous_upper = previous_bounds or ([5, 40, 60], [40, 255, 255])
        low_hue, high_hue = previous_lower[0] - 5, previous_upper[0] + 5
        if saturation >= 8:
            low_hue = min(low_hue, round(hue - 15))
            high_hue = max(high_hue, round(hue + 15))
        return ([max(0, low_hue), 8, 60], [min(179, high_hue), 255, 255])
    return ([max(0, round(hue - 10)), max(40, round(saturation - 100)),
             max(40, round(value - 140))],
            [min(179, round(hue + 10)), 255, 255])


def calibrate_camera(cap, cfg, config_path):
    points = []
    labels = ["LEFT reference (Q1)", "PIVOT", "RIGHT reference (Q8)", "BALL colour"]
    sample = [None]
    current_frame = [None]
    tracker = None
    trial = None

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            if len(points) == 4 and current_frame[0] is not None:
                patch = current_frame[0][max(0, y - 3):y + 4,
                                         max(0, x - 3):x + 4]
                if patch.size:
                    sample[0] = np.median(
                        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3),
                        axis=0)

    name = "Camera calibration"
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera did not return a frame")
            current_frame[0] = frame
            display = frame.copy()
            if len(points) == 4 and sample[0] is not None and trial is None:
                lower, upper = ball_colour_bounds(
                    sample[0], (cfg['ball_hsv_lower'], cfg['ball_hsv_upper']))
                pale = bool(cfg.get('ball_pale_mode', False) or sample[0][1] < 40)
                if pale:
                    # Clicking the rim must not discard the existing tolerance
                    # for a much paler rim elsewhere along the track.
                    lower[1], lower[2] = 8, 60
                trial = dict(cfg, ball_hsv_lower=lower, ball_hsv_upper=upper,
                             ball_pale_mode=pale, geometry=dict(zip(
                                 ('left_px', 'pivot_px', 'right_px'), points[:3])))
                tracker = BallTracker(trial)
            next_label = f"Click: {labels[len(points)]}" if len(points) < 4 else "Move ball to test; S save"
            cv2.putText(display, next_label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "B colour only / resample   R reset all   ESC quit", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
            for i, point in enumerate(points[:3]):
                cv2.circle(display, point, 5, (0, 255, 0), -1)
                label = ('LEFT', 'PIVOT', 'RIGHT')[i]
                label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                label_x = max(5, min(display.shape[1]-label_width-5, point[0]+8))
                label_y = max(90, min(display.shape[0]-10, point[1]+(22 if i == 1 else -12)))
                cv2.putText(display, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if tracker is not None:
                found = tracker.update(frame, time.monotonic())
                if found:
                    cv2.circle(display, found[2], found[3], (0, 255, 0), 2)
                status = f'Camera: {found[1]:+.1f} mm' if found else 'Camera: --'
                cv2.putText(display, status, (10, 73), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0) if found else (0, 100, 255), 2)
            if len(points) >= 3:
                cv2.arrowedLine(display, points[0], points[2], (255, 200, 0), 1, tipLength=0.03)
            cv2.imshow(name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord("r"):
                points.clear()
                sample[0] = None
                tracker = trial = None
            if key == ord('b'):
                geometry = cfg.get('geometry', {})
                if len(points) >= 3:
                    points[:] = points[:3]
                elif all(k in geometry for k in ('left_px', 'pivot_px', 'right_px')):
                    points[:] = [tuple(geometry[k]) for k in ('left_px', 'pivot_px', 'right_px')]
                if len(points) == 3:
                    sample[0] = None
                    tracker = trial = None
            if key == ord("s") and trial is not None:
                save_config(config_path, trial)
                cfg.update(trial)
                print("Camera calibration saved.")
                return True
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


# Tracking between frames

class BallTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.previous = None
        self.last_seen = None
        self.pending = None
        self.pending_at = None

    def update(self, frame, now):
        # Retain the search reference briefly across misses, but never return it
        # as a measurement. Sensor fallback/lost-ball handling still runs.
        if self.last_seen is not None and now - self.last_seen > 0.25:
            self.previous = None
            self.last_seen = None
            self.pending = None
            self.pending_at = None
        elapsed = now - self.last_seen if self.last_seen is not None else 0
        search_radius = min(90, 25 + 600 * max(0, elapsed))
        found = detect_ball(frame, self.cfg, self.previous, search_radius)
        if found is None:
            self.pending = None
            self.pending_at = None
            return None
        if self.previous is None:
            consistent = False
            if self.pending is not None and 0 < now - self.pending_at <= 0.15:
                distance = math.hypot(found[2][0] - self.pending[2][0],
                                      found[2][1] - self.pending[2][1])
                allowed = min(90, 25 + 600 * (now - self.pending_at))
                consistent = (distance <= allowed and
                              abs(found[3] - self.pending[3]) <= 0.25 * self.pending[3])
            if not consistent:
                self.pending, self.pending_at = found, now
                return None
        self.previous, self.last_seen = found, now
        self.pending = None
        self.pending_at = None
        return found
