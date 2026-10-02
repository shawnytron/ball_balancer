# Ball and beam balancer

## Typical operation

Run from the project folder with the environment active:

```sh
source .venv/bin/activate
```

Close Serial Monitor and any previous preview. Keep the camera and lighting fixed.
Replace `/dev/tty.usbmodem1301` below if your port changes
(`python -m serial.tools.list_ports` lists available ports).

### 1. Calibrate

```sh
beam-balancer --calibrate --camera 0 --port /dev/tty.usbmodem1301
```

The stages run automatically in order:

1. **Motor:** remove the ball. Type `start` + Enter, then use arrows to jog.
   Mark the lowest safe angle with `min`, the level beam with `level`, and the
   highest safe angle with `max`. Return to the marked level and type `save`.
   Choose limits before the linkage binds. Typed commands need Enter; arrows do not.
2. **Camera:** keep the beam level. Click **Q1 sensor centre → pivot → Q8 sensor
   centre → ball colour**. Use the first and last sensors, not the railing ends.
   Move the ball to check tracking, then press **S** to save. **R** restarts;
   **B** resamples colour while keeping geometry.
3. **Photoresistors:** remove the ball and press **Space** for the baseline.
   Follow the highlighted sensors Y0/Q1 through Y7/Q8, cover each with the ball,
   and press **Space**. Keep the ball and lighting steady during each capture.
   At the outer sensors, follow the displayed ball-centre position and magenta marker.

Each completed stage saves. `q` + Enter cancels motor calibration;
**Q / Esc** cancels a camera stage and stops the sequence. All nine photoresistor
captures must finish before that stage saves. Existing calibration can be reused
when the setup, ball and lighting have not changed.

### 2. Preview

```sh
beam-balancer --feedback fused --port /dev/tty.usbmodem1301
```

Move the ball slowly across the rail. Compare **Fused, Camera and Photoresistor**:
zero at the pivot, negative left, positive right. `--` means unavailable.
Preview sends no motor commands. Press **Q** to exit before continuing.

### 3. Balance

```sh
beam-balancer --run --port /dev/tty.usbmodem1301
```

Starts stopped at level. Set **Target = 0**, place the ball near the centre, then
click **Start**. The four sliders adjust **P, I, D and Target**.

- **Stop:** level the beam and clear the integral.
- **Save settings / S:** keep the four slider values for the next run.
- **Space:** toggle Start/Stop. **Q / Esc** or closing either window exits at level.

The plot shows 10 seconds of position (green) and target (blue), in mm.
Target changes clear the integral. Firmware returns to level after 400 ms without
servo commands.

## Pin layout — Elegoo Mega 2560

Use the PCB's pin-1 marking to identify connector orientation. Disconnect power
before changing wires.

| PCB J1 pin | Signal | Mega connection |
|---|---|---|
| 1 | +5 V | 5V |
| 2 | DA / MUX A | D22 |
| 3 | DB / MUX B | D24 |
| 4 | DC / MUX C | D26 |
| 5 | ADC_IN / MUX output | A0 |
| 6 | Ground | GND |

| Device connection | Connection |
|---|---|
| MG995 signal, usually orange/yellow | D28 |
| MG995 positive, red | Servo supply positive |
| MG995 ground, brown/black | Common GND with Mega and servo supply |
| Mega USB and webcam USB | Computer; camera index 0 |

The original setup powers the servo from Mega 5V. A loaded MG995 should use a
suitable separate regulated 5–6V supply with common ground; keep its positive
output separate from the USB-powered Mega's 5V rail. D28 uses the Servo library
and does not need a `~` PWM marking.

Y0–Y7 sensor positions are `[-84, -60, -36, -12, 12, 36, 60, 84]` mm from the pivot.
The **168 mm Q1–Q8 reference span** is `beam_length_mm` in
`config/beam_config.json`. The **205 mm inner rail** and **40 mm ball** give
ball-centre travel of **±82.5 mm**; targets allow **±80.5 mm** with 2 mm end clearance.

## Other commands

```sh
# Calibrate one component
beam-balancer --calibrate motor --port /dev/tty.usbmodem1301
beam-balancer --calibrate camera
beam-balancer --calibrate photoresistor --port /dev/tty.usbmodem1301

# Preview one measurement; all available values remain visible
beam-balancer --feedback camera
beam-balancer --feedback photoresistor --port /dev/tty.usbmodem1301

# Choose another camera or configuration file
beam-balancer --run --port /dev/tty.usbmodem1301 --camera 0 --config config/beam_config.json
beam-balancer --help
```

`--feedback` without a name selects fused. Add `--port` to camera preview to
include photoresistor measurements. All commands also accept
`python -m beam_balancer` or `python run.py` instead of `beam-balancer`.

## First-time setup

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m serial.tools.list_ports
```

### Arduino firmware

Requires Arduino AVR Boards and the Servo library. Close Serial Monitor and the
Python program before uploading. In Arduino IDE, open
`firmware/beam_balancer_arduino/beam_balancer_arduino.ino` and select
**Arduino Mega or Mega 2560 / ATmega2560**, or use:

```sh
export BEAM_ARDUINO_CLI="/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli"
"$BEAM_ARDUINO_CLI" board list
"$BEAM_ARDUINO_CLI" compile --fqbn arduino:avr:mega:cpu=atmega2560 --build-path /tmp/beam-balancer-build firmware/beam_balancer_arduino
"$BEAM_ARDUINO_CLI" upload --fqbn arduino:avr:mega:cpu=atmega2560 --port /dev/tty.usbmodem1301 --input-dir /tmp/beam-balancer-build --verify
```
