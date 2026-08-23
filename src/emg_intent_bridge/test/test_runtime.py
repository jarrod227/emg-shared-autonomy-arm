import queue
import time

import pytest

from emg_intent_bridge.runtime import DeviceClockMapper, SerialIntentReader


def anchored(**kwargs):
    """A mapper past its warm-up, anchored on a receipt with no buffering."""

    mapper = DeviceClockMapper(anchor_window=1, **kwargs)
    return mapper


def test_clock_mapper_preserves_device_intervals():
    mapper = anchored()

    first = mapper.map(1_000_000, 10_100_000_000)
    second = mapper.map(1_050_000, 10_150_000_000)

    assert second - first == 50_000_000


def test_clock_mapper_yields_no_stamp_until_the_warmup_window_closes():
    mapper = DeviceClockMapper(anchor_window=3)

    assert mapper.map(1_000_000, 10_100_000_000) is None
    assert mapper.map(1_050_000, 10_150_000_000) is None
    assert mapper.warming_up

    assert mapper.map(1_100_000, 10_200_000_000) is not None
    assert not mapper.warming_up


def test_clock_mapper_anchors_on_the_least_buffered_receipt_in_the_window():
    # The defect this replaced: anchoring on the first packet froze that
    # packet's buffering delay into every stamp for the life of the process.
    # A 0.13 s draw put every command past a consumer's 0.05 s future
    # tolerance, and 3913 consecutive commands were refused.
    mapper = DeviceClockMapper(anchor_window=3)

    # True host time of device time t is 9_100_000_000 + t; each receipt is
    # that plus its own buffering delay.
    mapper.map(1_000_000, 10_230_000_000)  # 130 ms buffered
    mapper.map(1_050_000, 10_230_000_000)  # 80 ms
    third = mapper.map(1_100_000, 10_300_000_000)  # 100 ms

    # Anchored on the 80 ms receipt, not the 130 ms one that arrived first.
    assert third == 9_180_000_000 + 1_100_000_000
    # The permanent forward bias is now the smallest delay in the window.
    assert third - (9_100_000_000 + 1_100_000_000) == 80_000_000


def test_clock_mapper_warms_up_again_after_a_reanchor():
    # A bad anchor is permanent, so the packet that merely reveals a jump is
    # no more trustworthy than the first packet was.
    mapper = DeviceClockMapper(max_skew_sec=1.0, anchor_window=2)
    mapper.map(1_000_000, 10_100_000_000)
    assert mapper.map(1_050_000, 10_150_000_000) is not None

    assert mapper.map(1_100_000, 25_000_000_000) is None
    assert mapper.reanchors == 1
    assert mapper.warming_up

    assert mapper.map(1_150_000, 25_050_000_000) is not None


def test_clock_mapper_rejects_a_window_smaller_than_one_packet():
    with pytest.raises(ValueError, match="anchor_window"):
        DeviceClockMapper(anchor_window=0)


def test_clock_mapper_never_steps_backwards_under_receipt_jitter():
    # A delayed first packet followed by a prompt one used to map the later
    # device time to an earlier ROS stamp. Freshness is judged on the stamp,
    # so the mapping must advance even when the tighter bound arrives late.
    mapper = anchored()

    delayed = mapper.map(1_000_000, 10_200_000_000)
    prompt = mapper.map(1_050_000, 10_150_000_000)

    assert delayed == 10_200_000_000
    assert prompt - delayed == 50_000_000


def test_clock_mapper_ignores_receipt_time_after_anchoring():
    # Two packets delivered in one host read share a receipt time. Tracking
    # the receipt would collapse the device interval between them, which is
    # the one thing the mapper must keep.
    mapper = anchored()

    mapper.map(1_000_000, 10_100_000_000)
    burst_first = mapper.map(1_500_000, 10_600_004_000)
    burst_second = mapper.map(2_000_000, 10_600_004_000)

    assert burst_first == 10_600_000_000
    assert burst_second - burst_first == 500_000_000


def test_clock_mapper_reanchors_after_a_clock_jump_and_counts_it():
    mapper = anchored(max_skew_sec=1.0)

    mapper.map(1_000_000, 10_100_000_000)
    jumped = mapper.map(1_050_000, 25_000_000_000)

    assert jumped == 25_000_000_000
    assert mapper.reanchors == 1
    assert mapper.map(1_100_000, 25_050_000_000) == 25_050_000_000


def test_clock_mapper_reset_forgets_anchor_and_monotonic_floor():
    mapper = anchored()
    mapper.map(1_000_000, 10_100_000_000)

    mapper.reset()

    assert not mapper.initialized
    assert mapper.map(2_000_000, 5_000_000_000) == 5_000_000_000
    assert mapper.reanchors == 0


def test_clock_mapper_rejects_a_nonpositive_skew_limit():
    with pytest.raises(ValueError, match="max_skew_sec"):
        DeviceClockMapper(max_skew_sec=0.0)


class FakeSerial:
    def __init__(self, *, chunks=(), **_kwargs):
        self._chunks = queue.Queue()
        for chunk in chunks:
            self._chunks.put(chunk)
        self.closed = False

    def read(self, _size):
        try:
            return self._chunks.get_nowait()
        except queue.Empty:
            time.sleep(0.001)
            return b""

    def close(self):
        self.closed = True

    def cancel_read(self):
        pass


def test_serial_reader_reports_open_failure_without_raising_in_caller():
    def fail(**_kwargs):
        raise OSError("port missing")

    reader = SerialIntentReader("/dev/missing", serial_factory=fail)
    reader.start()
    reader.stop()

    assert not reader.connected
    assert "port missing" in reader.error


def test_serial_reader_rejects_invalid_sizes():
    try:
        SerialIntentReader("x", timeout_sec=0)
    except ValueError as error:
        assert "timeout" in str(error)
    else:
        raise AssertionError("zero timeout accepted")
