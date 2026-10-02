"""Arduino serial protocol and fresh phototransistor samples."""
from collections import deque
import math
import threading
import time

import serial

from .config import validate_servo_limits


# Serial connection and servo commands

class ArduinoLink:
    def __init__(self, port):
        self.device = serial.Serial(port, 115200, timeout=0, write_timeout=0.25)
        self.buffer = bytearray()
        self.latest = None
        self.latest_at = 0.0
        self.latest_millis = None
        self.sample_count = 0
        self.responses = deque(maxlen=32)
        self.limits = None
        try:
            time.sleep(2.2)  # USB connection resets an UNO/Mega.
            self.device.reset_input_buffer()
            self.device.write(b"HELLO\n")
            self.wait_for("V,3")
        except Exception:
            self.device.close()
            raise

    def poll(self):
        incoming = self.device.read(self.device.in_waiting or 1)
        self.buffer.extend(incoming)
        if len(self.buffer) > 4096:
            self.buffer.clear()
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            try:
                message = line.strip().decode("ascii")
                fields = message.split(",")
                if fields[0] in ("V", "L", "ERR"):
                    self.responses.append(message)
                if len(fields) == 10 and fields[0] == "S":
                    device_millis = int(fields[1])
                    values = [int(value) for value in fields[2:]]
                    if 0 <= device_millis <= 0xFFFFFFFF and all(0 <= value <= 1023 for value in values):
                        self.latest = values
                        self.latest_at = time.monotonic()
                        self.latest_millis = device_millis
                        self.sample_count += 1
            except (ValueError, UnicodeDecodeError):
                pass

    def wait_for(self, expected, timeout=2.0):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            self.poll()
            while self.responses:
                message = self.responses.popleft()
                if message == expected:
                    return
                if expected.startswith("V,") and message.startswith("V,"):
                    raise RuntimeError("Arduino firmware " + message + " is incompatible; need " + expected +
                                       ". Upload the updated beam_balancer_arduino sketch for fractional servo control.")
                if message.startswith("ERR,"):
                    raise RuntimeError("Arduino rejected command: " + message)
            time.sleep(0.01)
        raise RuntimeError("Arduino did not confirm " + expected +
                           ". Upload the updated beam_balancer_arduino sketch "
                           "and close Serial Monitor before retrying.")

    def configure_limits(self, low, neutral, high):
        limits = validate_servo_limits(low, neutral, high)
        self.responses.clear()
        message = "L,{},{},{}".format(*limits)
        self.device.write((message + "\n").encode("ascii"))
        self.wait_for(message)
        self.limits = limits

    def angle(self, degrees):
        if self.limits is None:
            raise RuntimeError("Configure and confirm servo limits before movement")
        if not math.isfinite(degrees):
            raise ValueError("Servo angle must be finite")
        low, _, high = self.limits
        angle = max(low, min(high, degrees))
        # Firmware v3 converts fractional degrees to bounded 1-us servo pulses.
        self.device.write(f"A,{angle:.3f}\n".encode("ascii"))

    def stop(self):
        if self.limits is not None:
            self.device.write(b"STOP\n")

    def close(self):
        try:
            self.stop()
        finally:
            self.device.close()


# Background sensor sampling

class SensorReader:
    def __init__(self, link):
        self.link = link
        self.samples = deque(maxlen=128)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        sequence = self.link.sample_count
        try:
            while not self.stop.is_set():
                self.link.poll()
                if self.link.sample_count != sequence:
                    sequence = self.link.sample_count
                    sample = {"sequence": sequence, "received_at": self.link.latest_at,
                              "device_millis": self.link.latest_millis,
                              "adc": self.link.latest[:]}
                    with self.lock:
                        self.samples.append(sample)
                self.stop.wait(0.005)
        except Exception as exc:
            self.error = exc

    def nearest(self, received_at, max_age=0.06):
        if self.error:
            raise RuntimeError("Serial reader stopped: " + str(self.error))
        with self.lock:
            candidates = list(self.samples)
        if not candidates:
            return None
        result = min(candidates, key=lambda item: abs(item["received_at"] - received_at))
        return result if abs(result["received_at"] - received_at) <= max_age else None

    def between(self, start, end):
        """Fresh capture samples independent of camera frame rate."""
        if self.error:
            raise RuntimeError("Serial reader stopped: " + str(self.error))
        with self.lock:
            return [s for s in self.samples if start < s["received_at"] <= end]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=1)
