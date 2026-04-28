#!/usr/bin/env python3
"""
spi_test_sender.py -- Minimal SPI loopback test sender.

Sends [0xDE, 0xAD, 0xBE, 0xEF] once per second to the E1x EVK.
The EVK should print "rx: DE AD BE EF" on its STDIO console for each packet.

Usage:
    python spi_test_sender.py --ftdi-url ftdi://ftdi:232h:FTAD4WXF/1
"""

import argparse
import time
import sys

PAYLOAD = bytes([0xDE, 0xAD, 0xBE, 0xEF])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftdi-url", default="ftdi://ftdi:232h/1")
    parser.add_argument("--spi-freq", type=int, default=1_000_000)
    args = parser.parse_args()

    if sys.platform == 'win32':
        try:
            import libusb as _libusb
            import os as _os
            import usb.backend.libusb1 as _lb1
            _dll = _libusb.dll._name
            _os.add_dll_directory(_os.path.dirname(_dll))
            _lb1.get_backend(find_library=lambda _: _dll)
        except Exception:
            pass

    try:
        from pyftdi.spi import SpiController
    except ImportError:
        print("ERROR: pyftdi not found. pip install pyftdi")
        sys.exit(1)

    print(f"Opening {args.ftdi_url} at {args.spi_freq/1e6:.1f} MHz ...")
    ctrl = SpiController()
    ctrl.configure(args.ftdi_url)
    port = ctrl.get_port(cs=0, freq=args.spi_freq, mode=0)
    print("Ready. Sending DE AD BE EF every second — watch EVK console for 'rx: DE AD BE EF'")
    print("Press Ctrl+C to stop.")

    n = 0
    while True:
        port.write(PAYLOAD)
        print(f"  [{n}] sent: {' '.join(f'{b:02X}' for b in PAYLOAD)}")
        n += 1
        time.sleep(1.0)

if __name__ == "__main__":
    main()
