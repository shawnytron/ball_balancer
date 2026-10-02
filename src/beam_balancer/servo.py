"""Motor endpoint calibration and terminal controls."""
from contextlib import contextmanager
import os
import queue
import select
import sys
import threading
import time

from .config import config_servo_limits, save_config, validate_servo_limits


# Interactive terminal input

class KeyDecoder:
    """Immediate arrows plus editable text commands submitted with Enter."""

    def __init__(self, echo=None):
        self.escape = ""
        self.line = ""
        self.echo = echo or (lambda text: None)

    def feed(self, data):
        commands = []
        for char in data.decode("ascii", errors="ignore"):
            if self.escape:
                if self.escape == "\x1b":
                    self.escape = char if char in "[O" else ""
                elif "@" <= char <= "~":
                    # Only unmodified arrows; ignore other complete sequences.
                    if self.escape in ("[", "O") and char in "ABCD":
                        commands.append("+" if char in "AC" else "-")
                    self.escape = ""
                else:
                    self.escape += char
                continue
            if char == "\x1b":
                self.escape = char
            elif char in "\r\n":
                if self.line.strip():
                    commands.append(self.line.strip().lower())
                self.line = ""
                self.echo("\n")
            elif char in "\x7f\b":
                if self.line:
                    self.line = self.line[:-1]
                    self.echo("\b \b")
            elif char == "\x15":  # Ctrl-U clears the typed line.
                self.echo("\b \b" * len(self.line))
                self.line = ""
            elif char in "+-" and not self.line:
                commands.append(char)
            elif char == "\x04":  # Ctrl-D
                commands.append("q")
            elif char.isprintable() and len(self.line) < 80:
                self.line += char
                self.echo(char)
        return commands


@contextmanager
def calibration_commands():
    """Yield (poll, immediate); poll waits at most 30 ms for user input."""
    commands = queue.Queue()
    immediate = os.name == "posix" and sys.stdin.isatty()
    if immediate:
        import termios
        import tty

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        def echo(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        decoder = KeyDecoder(echo)

        def poll():
            if commands.empty() and select.select([fd], [], [], 0.03)[0]:
                data = os.read(fd, 64)
                for command in decoder.feed(data) if data else ["q"]:
                    commands.put(command)
            try:
                return commands.get_nowait()
            except queue.Empty:
                return None

        poll.pending_text = lambda: decoder.line

        try:
            # Keep Ctrl-C signals enabled; disable line buffering and echo.
            tty.setcbreak(fd)
            yield poll, True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
    else:
        def read_commands():
            try:
                while True:
                    command = input()
                    commands.put(command)
                    if command.strip().lower() == "q":
                        return
            except EOFError:
                commands.put("q")

        def poll():
            try:
                return commands.get(timeout=0.03)
            except queue.Empty:
                return None

        threading.Thread(target=read_commands, daemon=True).start()
        yield poll, False


# Motor calibration

class ServoCalibration:
    """Manual teaching state; no serial or file writes until explicitly requested."""
    def __init__(self, low, start, high):
        self.low, self.start, self.high = validate_servo_limits(low, start, high)
        self.angle = self.start
        self.started = False
        self.marks = {}

    def action(self, command):
        command = command.strip().lower()
        if command == "q":
            return "quit"
        if command == "start":
            self.started = True
            return "Holding position. Use the jog keys to move one degree."
        if not self.started:
            raise ValueError("Enable movement first at {} degrees".format(self.start))
        if command in ("+", "-"):
            self.angle = max(self.low, min(self.high,
                             self.angle + (1 if command == "+" else -1)))
            return "Commanded {} degrees".format(self.angle)
        if command in ("min", "level", "max"):
            self.marks[command] = self.angle
            return "Marked {} at {} degrees".format(command, self.angle)
        if command == "save":
            if set(self.marks) != {"min", "level", "max"}:
                raise ValueError("Mark min, level, and max before saving")
            validate_servo_limits(self.marks["min"], self.marks["level"], self.marks["max"])
            if self.angle != self.marks["level"]:
                raise ValueError("Jog back to your marked level ({} degrees) before saving"
                                 .format(self.marks["level"]))
            return "save"
        raise ValueError("Commands: start, +, -, min, level, max, save, q")

    def saved_config(self, cfg):
        if self.action("save") != "save":
            raise ValueError("Calibration is incomplete")
        result = dict(cfg)
        result.update(servo_min_deg=self.marks["min"],
                      servo_neutral_deg=self.marks["level"],
                      servo_max_deg=self.marks["max"], servo_calibrated=True)
        return result


def calibrate_servo(link, cfg, config_path):
    state = ServoCalibration(5, cfg["servo_neutral_deg"], 85)
    link.configure_limits(state.low, state.start, state.high)
    print("Linkage attached, ball removed. Choose operating limits BEFORE binding.")
    print("This records commanded angles; it cannot sense a mechanical stop.")
    print("Jog range: {}..{} deg; initial command: {} deg."
          .format(state.low, state.high, state.start))
    print("On exit/link loss the servo returns to the session's level.")
    print("Before saving, the fallback level is {} deg. Enable only if this is safe."
          .format(state.start))
    with calibration_commands() as (read_command, immediate):
        if immediate:
            print("Arrows or +/-: jog 1 degree immediately (no Enter)")
            print("Type start, min, level, max, save, or q, then press Enter.")
            print("Tap arrows for individual steps; holding repeats. Backspace edits text.")
        else:
            print("Line mode: start, +, -, min, level, max, save, q (Enter after each)")
        try:
            return _teach_servo(link, cfg, config_path, state, read_command)
        finally:
            if state.started:
                link.stop()


def _teach_servo(link, cfg, config_path, state, read_command):
    last_send = 0.0
    print("Command > ", end="", flush=True)
    while True:
        link.poll()
        if state.started and time.monotonic() - link.latest_at > 0.5:
            raise RuntimeError("Sensor telemetry stopped; ending calibration")
        command = read_command()
        if command is not None:
            try:
                result = state.action(command)
                if result == "quit":
                    print("Calibration cancelled; saved limits were not changed.")
                    return False
                if result == "save":
                    updated = state.saved_config(cfg)
                    link.configure_limits(*config_servo_limits(updated))
                    save_config(config_path, updated)
                    cfg.update(updated)
                    print("Saved minimum={}, level={}, maximum={} degrees to {}"
                          .format(*config_servo_limits(updated), config_path))
                    return True
                print(result)
            except ValueError as exc:
                print(exc)
            pending = getattr(read_command, "pending_text", lambda: "")()
            print("At {} deg | marks {} | Command > {}".format(state.angle, state.marks, pending),
                  end="", flush=True)
        now = time.monotonic()
        if state.started and (command is not None or now - last_send >= 0.1):
            if now - link.latest_at > 0.5:
                raise RuntimeError("No fresh Arduino telemetry; servo command cancelled")
            link.angle(state.angle)
            last_send = now
