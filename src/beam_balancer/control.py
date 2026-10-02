"""Position PID, actuator bounds and the interactive tuning panel."""
from collections import deque
import math

import cv2
import numpy as np

from .config import ball_travel_bounds, config_servo_limits, load_config, save_config


# Controller settings and PID

PID_KEYS = ('kp_beam_deg_per_mm', 'ki_beam_deg_per_mm_s',
            'kd_beam_deg_per_mm_s', 'setpoint_mm')


def target_bounds(cfg):
    low, high = ball_travel_bounds(cfg)
    margin = cfg.get('setpoint_margin_mm', 2.0)
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or margin < 0 or high-low <= 2*margin):
        raise ValueError('setpoint_margin_mm must leave room for the ball to travel')
    return low+margin, high-margin


def validate_pid_settings(cfg):
    config_servo_limits(cfg)
    defaults = {'kp_beam_deg_per_mm': .025, 'ki_beam_deg_per_mm_s': 0,
                'kd_beam_deg_per_mm_s': .008, 'integral_limit_beam_deg': 1.0,
                'derivative_filter_tau_s': .1, 'max_servo_rate_deg_s': 25.0,
                'loss_hold_s': .10, 'loss_return_rate_deg_s': 25.0,
                'reacquire_stable_s': .06,
                'max_beam_tilt_deg': 3.0, 'servo_deg_per_beam_deg': 2.775,
                'position_deadband_mm': 0.0,
                'setpoint_mm': 0.0}
    for key, default in defaults.items():
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if key != 'setpoint_mm' and value < 0:
            raise ValueError(f'{key} must be nonnegative')
    if cfg.get('max_servo_rate_deg_s', 25) <= 0 or cfg.get('servo_deg_per_beam_deg', 2.775) <= 0:
        raise ValueError('Servo rate and servo-to-beam ratio must be positive')
    if cfg.get('loss_return_rate_deg_s', 25) <= 0:
        raise ValueError('Loss return rate must be positive')
    if cfg.get('loss_hold_s', .10) > .2 or cfg.get('reacquire_stable_s', .06) > .2:
        raise ValueError('Loss hold and reacquisition times must be at most 0.2 seconds')
    if cfg.get('servo_sign') not in (-1, 1):
        raise ValueError('servo_sign must be +1 or -1')
    low, high = target_bounds(cfg)
    if not low <= cfg.get('setpoint_mm', 0) <= high:
        raise ValueError(f'Setpoint must be within {low:g}..{high:g} mm')


class BalanceController:
    """Positive beam tilt raises camera-right; use the calibrated servo sign."""
    def __init__(self, cfg):
        validate_pid_settings(cfg)
        self.cfg = cfg
        self.last_measurement = None
        self.last_time = None
        self.last_output_time = None
        self.velocity = 0.0
        self.servo_angle = cfg['servo_neutral_deg']
        self.integral_term = 0.0  # Stored in beam degrees; gain edits don't rescale it.
        self.p_term = self.d_term = self.beam_tilt = 0.0
        self.limited = self.integrator_held = False
        self.target = cfg['setpoint_mm']
        self.control_state = 'level'
        self.last_feedback_at = None
        self.loss_active = False
        self.loss_age_s = None
        self.reacquire_since = None
        self.reacquire_count = 0
        self.position_deadband_active = False
        self.effective_displacement_mm = None

    def reset_integral(self):
        self.integral_term = 0.0
        self.integrator_held = False

    def reset_velocity(self):
        """Clear derivative history on a measurement-source change."""
        self.last_measurement = self.last_time = None
        self.velocity = 0.0

    def level(self):
        """Explicit pause/stop response: neutral immediately, clearing all state."""
        self.reset_integral()
        self.reset_velocity()
        self.last_output_time = None
        self.p_term = self.d_term = self.beam_tilt = 0.0
        self.limited = False
        self.servo_angle = self.cfg['servo_neutral_deg']
        self.control_state = 'level'
        self.last_feedback_at = None
        self.loss_active = False
        self.loss_age_s = None
        self.reacquire_since = None
        self.reacquire_count = 0
        self.position_deadband_active = False
        self.effective_displacement_mm = None
        return self.servo_angle

    def _begin_loss(self):
        self.loss_active = True
        self.reacquire_since = None
        self.reacquire_count = 0
        self.reset_integral()
        self.reset_velocity()
        self.p_term = self.d_term = 0.0
        self.position_deadband_active = False
        self.effective_displacement_mm = None

    def _loss_output(self, now):
        """Hold a bounded command briefly, then slew to neutral without PID."""
        cfg = self.cfg
        self.reset_integral()
        self.p_term = self.d_term = 0.0
        self.loss_age_s = now-self.last_feedback_at if self.last_feedback_at is not None else None
        hold = cfg.get('loss_hold_s', .10)
        if self.loss_age_s is not None and self.loss_age_s <= hold:
            self.control_state = 'holding'
        else:
            dt = now-self.last_output_time if self.last_output_time is not None else .04
            if not 0 < dt < .3:
                dt = .04
            # A frame straddling the hold deadline only uses the time after it.
            if self.last_feedback_at is not None:
                dt = min(dt, max(0.0, now-self.last_feedback_at-hold))
            rate = min(cfg.get('loss_return_rate_deg_s', 25), cfg.get('max_servo_rate_deg_s', 25))
            delta = cfg['servo_neutral_deg']-self.servo_angle
            self.servo_angle += max(-rate*dt, min(rate*dt, delta))
            self.control_state = ('lost_level' if self.servo_angle == cfg['servo_neutral_deg']
                                  else 'returning')
        self.last_output_time = now
        self.beam_tilt = ((self.servo_angle-cfg['servo_neutral_deg']) /
                          (cfg['servo_sign']*cfg['servo_deg_per_beam_deg']))
        self.limited = self.integrator_held = False  # PID is inactive during loss.
        return self.servo_angle

    def update(self, position, now):
        cfg = self.cfg
        if not math.isfinite(now) or (self.last_output_time is not None and now < self.last_output_time):
            return self.level()
        gap = now-self.last_output_time if self.last_output_time is not None else 0.0
        if gap >= .4:
            # The v3 board's watchdog will have commanded neutral during a host
            # stall. Resume from neutral and require stable fresh feedback.
            self.level()
            self._begin_loss()
        elif gap >= .3:
            self._begin_loss()
        if self.target != cfg['setpoint_mm']:
            self.reset_integral()
            self.target = cfg['setpoint_mm']
        if position is None or not math.isfinite(position):
            if not self.loss_active:
                self._begin_loss()
            self.reacquire_since = None
            self.reacquire_count = 0
            return self._loss_output(now)
        if self.loss_active:
            if self.reacquire_since is None:
                self.reacquire_since = now
            self.reacquire_count += 1
            # Isolated detections do not restart the hold deadline indefinitely.
            if self.reacquire_count < 2 or now-self.reacquire_since < cfg.get('reacquire_stable_s', .06):
                return self._loss_output(now)
            self.loss_active = False
            self.reacquire_since = None
            self.reacquire_count = 0
            self.reset_velocity()  # No derivative or integral across the gap.
        self.control_state = 'tracking'
        self.last_feedback_at = now
        self.loss_age_s = 0.0
        dt = now-self.last_time if self.last_time is not None else 0.0
        output_dt = now-self.last_output_time if self.last_output_time is not None else .04
        if dt < 0 or dt >= .3:
            self.reset_velocity()
            self.reset_integral()
            dt = 0.0
        if not 0 < output_dt < .3:
            output_dt = .04
        if self.last_measurement is not None and dt > 0:
            raw_velocity = (position-self.last_measurement)/dt
            tau = cfg.get('derivative_filter_tau_s', .1)
            alpha = dt/(tau+dt)
            self.velocity += alpha*(raw_velocity-self.velocity)
        self.last_measurement, self.last_time = position, now
        self.last_output_time = now
        displacement = position-cfg['setpoint_mm']
        deadband = cfg.get('position_deadband_mm', 0.0)
        self.position_deadband_active = deadband > 0 and abs(displacement) <= deadband
        # A continuous dead zone: remove the tolerated position error from P
        # and I. D still sees real ball velocity and can brake through target.
        self.effective_displacement_mm = math.copysign(max(0.0, abs(displacement)-deadband), displacement)
        self.p_term = cfg['kp_beam_deg_per_mm']*self.effective_displacement_mm
        self.d_term = cfg['kd_beam_deg_per_mm_s']*self.velocity
        ki = cfg.get('ki_beam_deg_per_mm_s', 0.0)
        if ki == 0:
            self.reset_integral()
        # Zero increment inside the band retains any learned levelling bias.
        increment = ki*self.effective_displacement_mm*dt
        limit_i = min(cfg.get('integral_limit_beam_deg', 1.0), cfg['max_beam_tilt_deg'])
        candidate_i = max(-limit_i, min(limit_i, self.integral_term+increment))
        neutral = cfg['servo_neutral_deg']
        conversion = cfg['servo_sign']*cfg['servo_deg_per_beam_deg']
        step = cfg.get('max_servo_rate_deg_s', 25.0)*output_dt

        def constrain(tilt):
            tilt = max(-cfg['max_beam_tilt_deg'], min(cfg['max_beam_tilt_deg'], tilt))
            target = max(cfg['servo_min_deg'], min(cfg['servo_max_deg'], neutral+conversion*tilt))
            servo = self.servo_angle+max(-step, min(step, target-self.servo_angle))
            servo = max(cfg['servo_min_deg'], min(cfg['servo_max_deg'], servo))
            return servo, (servo-neutral)/conversion

        requested = self.p_term+candidate_i+self.d_term
        servo, applied = constrain(requested)
        # Conditional integration accounts for beam, mechanical, AND slew limits.
        # Still allow the integrator to unwind toward the achievable command.
        self.integrator_held = (requested-applied)*increment > 1e-10
        if self.integrator_held:
            available_i = applied-self.p_term-self.d_term
            # Allow the achievable part of an I increment. Fully freezing I
            # during slew limiting could prevent an I-only command ever moving.
            if increment > 0:
                candidate_i = max(self.integral_term, min(candidate_i, available_i))
            else:
                candidate_i = min(self.integral_term, max(candidate_i, available_i))
            self.integral_term = candidate_i
            requested = self.p_term+self.integral_term+self.d_term
            servo, applied = constrain(requested)
        else:
            self.integral_term = candidate_i
        self.limited = abs(requested-applied) > 1e-8
        self.beam_tilt = applied
        self.servo_angle = servo
        return servo


# Live tuning panel

class PIDPanel:
    name = 'PID controls'
    width, height = 860, 600

    def __init__(self, cfg, controller, config_path):
        self.cfg, self.controller, self.config_path = cfg, controller, config_path
        self.paused = True
        self.quit = False
        self.dragging = None
        self.history = deque(maxlen=1500)
        self.notice = ''
        low, high = target_bounds(cfg)
        self.plot_span = math.ceil((max(abs(low), abs(high)) + cfg.get('setpoint_margin_mm', 2) + 5)/10)*10
        self.rows = [
            ('kp_beam_deg_per_mm', 'P', 'beam deg / mm', 0, max(.15, cfg['kp_beam_deg_per_mm']*2), .001, 3),
            ('ki_beam_deg_per_mm_s', 'I', 'beam deg / (mm s)', 0, max(.020, cfg.get('ki_beam_deg_per_mm_s', 0)*2), .0001, 4),
            ('kd_beam_deg_per_mm_s', 'D', 'beam deg s / mm', 0, max(.050, cfg['kd_beam_deg_per_mm_s']*2), .0001, 4),
            ('setpoint_mm', 'Target', 'mm from pivot', low, high, .5, 1),
        ]
        self.buttons = [('start', 'Start', 24, 270), ('stop', 'Stop', 286, 532),
                        ('save', 'Save settings [S]', 548, 836)]

    def open(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.name, self.mouse)

    def close(self):
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass

    def set_value(self, row_index, value):
        key, _, _, low, high, step, _ = self.rows[row_index]
        self.cfg[key] = round(max(low, min(high, round(value/step)*step)), 6)
        self.notice = ''
        if key == 'setpoint_mm' or (key == 'ki_beam_deg_per_mm_s' and self.cfg[key] == 0):
            self.controller.reset_integral()

    def action(self, action):
        if action in ('start', 'stop'):
            # Repeated Start presses must not keep resetting a running PID.
            paused = action == 'stop'
            if paused or self.paused:
                self.controller.level()
            self.paused = paused
        elif action == 'save':
            try:
                saved = load_config(self.config_path)
                saved.update({key: self.cfg.get(key, 0.0) for key in PID_KEYS})
                validate_pid_settings(saved)
                save_config(self.config_path, saved)
                self.notice = 'Settings saved'
            except (OSError, ValueError) as exc:
                self.notice = 'Save failed: ' + str(exc)
        elif action == 'quit':
            self.paused = self.quit = True
            self.controller.level()

    def mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = None
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, _, left, right in self.buttons:
                if left <= x <= right and 325 <= y <= 367:
                    self.action(action)
                    return
            for i in range(len(self.rows)):
                if abs(y-(90+60*i)) <= 17 and 230 <= x <= 720:
                    self.dragging = i
                    break
        if self.dragging is not None and (event == cv2.EVENT_LBUTTONDOWN or flags & cv2.EVENT_FLAG_LBUTTON):
            row = self.rows[self.dragging]
            fraction = max(0, min(1, (x-230)/490))
            self.set_value(self.dragging, row[3]+fraction*(row[4]-row[3]))

    def handle_key(self, key):
        if key == 32:
            self.action('start' if self.paused else 'stop')
        elif key == ord('s'):
            self.action('save')
        elif key in (ord('q'), 27):
            self.action('quit')

    def render(self, now, position):
        self.history.append((now, position, self.cfg['setpoint_mm']))
        while self.history and now-self.history[0][0] > 10:
            self.history.popleft()
        frame = np.full((self.height, self.width, 3), (25, 23, 22), np.uint8)
        white, muted, green, blue = (238, 238, 238), (160, 158, 153), (129, 223, 112), (250, 182, 83)

        def text(value, x, y, size=.48, colour=white):
            cv2.putText(frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)

        text('PID controls', 24, 35, .8)
        text('Stopped' if self.paused else 'Running', 710, 34, .55, muted if self.paused else green)
        for i, row in enumerate(self.rows):
            key, label, unit, low, high, _, digits = row
            y = 90+60*i
            value = self.cfg.get(key, 0)
            text(label, 24, y-2, .60)
            text(unit, 24, y+18, .36, muted)
            cv2.line(frame, (230, y), (720, y), (75, 72, 69), 5)
            knob = int(230+490*max(0, min(1, (value-low)/(high-low))))
            cv2.line(frame, (230, y), (knob, y), blue, 5)
            cv2.circle(frame, (knob, y), 8, white, -1, cv2.LINE_AA)
            text(f'{low:g}', 230, y+23, .34, muted)
            text(f'{high:g}', 685, y+23, .34, muted)
            text(f'{value:+.{digits}f}' if key == 'setpoint_mm' else f'{value:.{digits}f}', 752, y+5, .5, blue)
        for action, label, left, right in self.buttons:
            colour = (65, 58, 49)
            if (action == 'stop' and self.paused) or (action == 'start' and not self.paused):
                colour = (65, 86, 49)
            cv2.rectangle(frame, (left, 325), (right, 367), colour, -1)
            text(label, left+18, 353, .55)
        text(self.notice[:100], 24, 395, .4, muted)
        self._plot(frame, now)
        text('10 s   |   position (green)   target (blue)   |   mm', 60, 566, .40, muted)
        return frame

    def _plot(self, frame, now):
        left, right, top, bottom = 60, 835, 420, 542
        span = self.plot_span  # Fixed scale: labels do not jump as the ball moves.
        for tick in (-span, 0, span):
            y = round(bottom-(tick+span)/(2*span)*(bottom-top))
            cv2.line(frame, (left, y), (right, y), (58, 54, 51), 1)
            cv2.putText(frame, f'{tick:+g}', (10, y+4), cv2.FONT_HERSHEY_SIMPLEX, .35, (160, 160, 160), 1)
        for col, colour in ((1, (129, 223, 112)), (2, (250, 182, 83))):
            previous = previous_at = None
            for row in self.history:
                value = row[col]
                if value is None or not math.isfinite(value):
                    previous = previous_at = None
                    continue
                point = (round(right-(now-row[0])/10*(right-left)),
                         round(bottom-(max(-span, min(span, value))+span)/(2*span)*(bottom-top)))
                if previous is not None and row[0]-previous_at < .3:
                    cv2.line(frame, previous, point, colour, 2, cv2.LINE_AA)
                previous, previous_at = point, row[0]

    def show(self, now, position):
        cv2.imshow(self.name, self.render(now, position))
        if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            self.action('quit')
