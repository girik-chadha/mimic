"""Bridge between WSL and the arm's USB serial port. Runs on WINDOWS Python (WSL cannot see USB).

Started from WSL by mimic.arm.SerialBridgeTransport, roughly:
    py.exe -u <this file, as a Windows path> --port auto

It prints READY once the port is open, then for every command line it reads on stdin it writes the
line to the arm at 115200 baud and prints the arm's one-line reply (or TIMEOUT). Lines from the
ESP32 that are not replies (boot messages) go to stderr as "ignored: ...". When stdin closes, or on
any error, or on Ctrl+C, it sends S (stop) and exits. If the port cannot be opened it prints "ERROR <reason>".

Only the standard library and pyserial: Windows Python does not have the rest of the project.
--fake pretends to be a locked arm, so this script can be tested without hardware.
"""
import argparse
import sys
import time

BAUD = 115200
REPLY_TIMEOUT_S = 2.0
BOOT_WAIT_S = 2.0           # opening the port may reset the ESP32; it boots in well under 2 s
CP210X = (0x10C4, 0xEA60)   # the USB chip on the arm's ESP32 board (needs.md A7b)


def is_reply(line: str) -> bool:
    return line.startswith(("OK", "ERR", "A "))


# ---------- choosing and opening the port ----------

def choose_port(wanted: str, ports: list) -> str:
    """ports: (device, vid, pid) for every serial port. Only the arm's CP210x chip is ever accepted,
    so a Bluetooth port (COM5/COM6 on this laptop) can never be opened by mistake."""
    arm_ports = [device for device, vid, pid in ports if (vid, pid) == CP210X]
    if wanted == "auto":
        if len(arm_ports) != 1:
            raise RuntimeError(f"found {len(arm_ports)} CP210x ports {arm_ports}; plug in the arm only, or pass --port")
        return arm_ports[0]
    if wanted not in arm_ports:
        raise RuntimeError(f"{wanted} is not the arm's CP210x port (CP210x ports: {arm_ports or 'none'})")
    return wanted


def open_port(wanted: str, boot_wait_s: float):
    import serial  # only on Windows, where pyserial is installed
    from serial.tools import list_ports

    name = choose_port(wanted, [(p.device, p.vid, p.pid) for p in list_ports.comports()])
    port = serial.Serial()
    port.port = name
    port.baudrate = BAUD
    port.timeout = 0.1
    port.dtr = False  # set before opening: DTR and RTS low keep the ESP32 from resetting...
    port.rts = False
    port.open()
    time.sleep(boot_wait_s)   # ...and if it resets anyway, give it time to boot,
    port.reset_input_buffer()  # then throw away its boot messages
    return port


class FakePort:
    """Pretends to be the locked firmware: J and H are refused, ? ends with L. 'silent' never replies.
    Writes every line it is sent to `log_path`, so tests can check that S was sent at the end."""

    def __init__(self, mode="locked", log_path=None):
        self.mode = mode
        self.log_path = log_path
        self.waiting = [b"rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)\n"]  # boot text
        self.first_reply = True

    def write(self, data: bytes):
        line = data.decode().strip()
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        if self.mode == "silent":
            return
        if self.first_reply:  # an ESP32 log line before the first reply, to exercise "ignored"
            self.waiting.append(b"E (512) some log message\n")
            self.first_reply = False
        if line == "?":
            reply = "A 90 90 90 90 L"
        elif line == "S":
            reply = "OK"
        elif line == "H" or line.startswith("J "):
            reply = "ERR limits not measured"
        else:
            reply = "ERR bad command"
        self.waiting.append(reply.encode() + b"\n")

    def readline(self) -> bytes:
        if self.waiting:
            return self.waiting.pop(0)
        time.sleep(0.02)
        return b""

    def reset_input_buffer(self):
        self.waiting.clear()

    def close(self):
        pass


# ---------- the bridge ----------

def exchange(port, line: str, err, reply_timeout_s: float) -> str:
    """Send one command and return the arm's reply line, or TIMEOUT."""
    port.write((line + "\n").encode())
    deadline = time.monotonic() + reply_timeout_s
    pending = b""
    while time.monotonic() < deadline:
        pending += port.readline()
        if not pending.endswith(b"\n"):
            continue  # nothing yet, or half a line: keep reading
        text = pending.decode(errors="replace").strip()
        pending = b""
        if is_reply(text):
            return text
        if text:
            err.write(f"ignored: {text}\n")
    return "TIMEOUT"


def bridge(port, lines, out, err, reply_timeout_s=REPLY_TIMEOUT_S) -> None:
    out.write("READY\n")
    out.flush()
    try:
        for line in lines:
            line = line.strip()
            if line:
                out.write(exchange(port, line, err, reply_timeout_s) + "\n")
                out.flush()
    finally:  # stdin closed, or something went wrong: always tell the arm to stop
        try:
            exchange(port, "S", err, 0.5)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="WSL <-> arm serial bridge (run with Windows Python).")
    parser.add_argument("--port", default="auto", help="COM port of the arm's CP210x chip, or auto")
    parser.add_argument("--boot-wait", type=float, default=BOOT_WAIT_S)
    parser.add_argument("--reply-timeout", type=float, default=REPLY_TIMEOUT_S)
    parser.add_argument("--fake", choices=["locked", "silent", "broken"], help="no hardware: pretend to be an arm")
    parser.add_argument("--fake-log", help="with --fake: file that records every line sent to the fake arm")
    args = parser.parse_args()
    try:
        if args.fake == "broken":
            raise RuntimeError("could not open port (fake)")
        port = FakePort(args.fake, args.fake_log) if args.fake else open_port(args.port, args.boot_wait)
    except Exception as error:
        print(f"ERROR {error}", flush=True)
        sys.exit(1)
    try:
        bridge(port, iter(sys.stdin.readline, ""), sys.stdout, sys.stderr, args.reply_timeout)
    except KeyboardInterrupt:  # Ctrl+C: bridge() has already sent S on its way out
        sys.exit(130)
    finally:
        port.close()


if __name__ == "__main__":
    main()
