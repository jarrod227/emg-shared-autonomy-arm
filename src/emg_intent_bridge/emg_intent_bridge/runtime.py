"""Runtime primitives for the serial bridge, kept independent from rclpy."""

from dataclasses import dataclass
import queue
import threading
import time

import serial

from .confirmation import DeviceIntent
from .intent_stream import IntentStreamDecoder


@dataclass(frozen=True)
class ReceivedIntent:
    """One decoded intent plus its host monotonic receipt time."""

    intent: DeviceIntent
    received_monotonic_ns: int


class DeviceClockMapper:
    """Map unwrapped MCU microseconds onto the active ROS clock.

    The absolute offset is unknowable over a byte stream: each receipt only
    bounds source time from above. An earlier version kept the smallest
    observed ``receipt_ros - device_time``, which is the tightest bound but
    lets a later, less-buffered packet map to an *earlier* ROS stamp than one
    that already went out. Consumers judge freshness on the stamp, so a stream
    that steps backwards can retire a newer command in favour of an older one.

    Monotonicity therefore outranks tightness. The first receipt anchors the
    mapping and every later stamp is that anchor plus the MCU's own elapsed
    time. Chasing a tighter offset was tried and removed: when two packets
    arrive in one host read they share a receipt time, so tightening tracks
    host buffering instead of the device and collapses the 50 ms source grid
    the mapper exists to preserve. Freezing costs a constant bias equal to the
    first packet's buffering delay, which is bounded by the caller's own
    receipt-age check, and buys exact intervals.

    A jump larger than ``max_skew_ns`` means the anchor no longer describes
    reality (ROS time was set, or the MCU restarted); the mapper re-anchors
    and counts it so the caller can discard evidence that straddles the
    discontinuity.
    """

    def __init__(self, *, max_skew_sec=1.0):
        if max_skew_sec <= 0.0:
            raise ValueError("max_skew_sec must be positive")
        self.max_skew_ns = round(float(max_skew_sec) * 1e9)
        self.reanchors = 0
        self._offset_ns = None
        self._last_mapped_ns = None

    @property
    def initialized(self):
        return self._offset_ns is not None

    def reset(self):
        self._offset_ns = None
        self._last_mapped_ns = None

    def map(self, device_timestamp_us, receipt_ros_ns):
        device_ns = int(device_timestamp_us) * 1000
        candidate = int(receipt_ros_ns) - device_ns
        if self._offset_ns is None:
            self._offset_ns = candidate
        elif abs(candidate - self._offset_ns) > self.max_skew_ns:
            self.reanchors += 1
            self._offset_ns = candidate
            self._last_mapped_ns = None

        mapped = max(0, self._offset_ns + device_ns)
        if self._last_mapped_ns is not None:
            mapped = max(mapped, self._last_mapped_ns)
        self._last_mapped_ns = mapped
        return mapped


class SerialIntentReader:
    """Own the serial port and decode it away from the ROS executor thread."""

    def __init__(self, port, *, baudrate=115200, timeout_sec=0.05,
                 queue_size=512, read_size=4096, serial_factory=None):
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        if queue_size <= 0 or read_size <= 0:
            raise ValueError("queue_size and read_size must be positive")
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout_sec = float(timeout_sec)
        self.read_size = int(read_size)
        self.decoder = IntentStreamDecoder()
        self._queue = queue.Queue(maxsize=int(queue_size))
        self._serial_factory = serial.Serial if serial_factory is None else serial_factory
        self._stop = threading.Event()
        self._thread = None
        self._connection = None
        self.connected = False
        self.error = None
        self.queue_drops = 0
        self.bytes_received = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("serial reader already started")
        self._thread = threading.Thread(
            target=self._run,
            name="emg-intent-serial",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        try:
            self._connection = self._serial_factory(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout_sec,
            )
            self.connected = True
            while not self._stop.is_set():
                chunk = self._connection.read(self.read_size)
                if not chunk:
                    continue
                received_ns = time.monotonic_ns()
                self.bytes_received += len(chunk)
                for intent in self.decoder.feed(chunk):
                    try:
                        self._queue.put_nowait(ReceivedIntent(intent, received_ns))
                    except queue.Full:
                        # Drop newest. The ROS side watches this counter and
                        # invalidates any half-complete confirmation pair.
                        self.queue_drops += 1
        except Exception as error:  # serial backends use several error types
            if not self._stop.is_set():
                self.error = f"{type(error).__name__}: {error}"
        finally:
            self.connected = False
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def pop_nowait(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def write(self, data):
        """Send bytes to the device. Returns False if nothing is open yet.

        Safe to call from a different thread than _run(): on POSIX, pyserial
        read() and write() are independent os.read()/os.write() calls on the
        same fd, so one thread reading while another writes needs no lock --
        the two directions do not share mutable state.
        """
        connection = self._connection
        if connection is None:
            return False
        try:
            connection.write(data)
            return True
        except Exception as error:  # serial backends use several error types
            self.error = f"{type(error).__name__}: {error}"
            return False

    def stop(self):
        self._stop.set()
        connection = self._connection
        if connection is not None:
            cancel = getattr(connection, "cancel_read", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.timeout_sec))

