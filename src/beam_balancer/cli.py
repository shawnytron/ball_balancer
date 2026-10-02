"""Calibrate the balancer, preview measurements, or run the PID panel."""
import argparse
from pathlib import Path

import cv2
import serial

from .arduino import ArduinoLink
from .config import CONFIG_PATH, load_config
from .control import validate_pid_settings
from .runtime import run
from .sensors import calibrate_sensors
from .servo import calibrate_servo
from .vision import calibrate_camera


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--calibrate', nargs='?', const='all',
                       choices=('camera', 'motor', 'photoresistor', 'all'), metavar='PART',
                       help='camera, motor, or photoresistor; omit PART to calibrate all three')
    modes.add_argument('--feedback', nargs='?', const='fused',
                       choices=('camera', 'photoresistor', 'fused'), metavar='SOURCE',
                       help='Preview camera, photoresistor, or fused measurements without motor commands')
    modes.add_argument('--run', action='store_true', help='Open PID controls, stopped at level')
    parser.add_argument('--port', help='Arduino serial port, e.g. /dev/cu.usbmodem113401')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    needs_port = (args.run or args.calibrate in ('all', 'motor', 'photoresistor')
                  or args.feedback in ('fused', 'photoresistor'))
    if needs_port and not args.port:
        parser.error('This mode requires --port')
    cfg = load_config(args.config)
    if args.run:
        validate_pid_settings(cfg)
        if cfg.get('servo_calibrated') is not True:
            parser.error('Run --calibrate motor and save the operating limits first')

    link = cap = None
    try:
        if args.port:
            link = ArduinoLink(args.port)
        steps = ('motor', 'camera', 'photoresistor') if args.calibrate == 'all' else (args.calibrate,)
        for step in steps:
            if step == 'motor':
                if not calibrate_servo(link, cfg, args.config):
                    return
                continue
            if cap is None:
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    raise RuntimeError(f'Cannot open camera index {args.camera}')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if step == 'camera':
                completed = calibrate_camera(cap, cfg, args.config)
            elif step == 'photoresistor':
                completed = calibrate_sensors(link, cfg, args.config, cap)
            else:
                run(cap, link, cfg, config_path=args.config, preview=args.feedback)
                return
            if not completed:
                print('Calibration cancelled; completed stages remain saved.')
                return
    finally:
        try:
            if link:
                link.close()
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()


def entrypoint():
    """Shared console-script, module, and checkout entry point."""
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except (OSError, RuntimeError, ValueError, serial.SerialException) as exc:
        raise SystemExit(str(exc))
