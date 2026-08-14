#!/usr/bin/env python3
"""GUI-guided, automatically labelled Objective 3.5 sEMG collection.

The window previews all three raw ADC channels, checks electrode contact, and
guides a balanced randomized session through these accepted gestures:

    REST          -> RELAX
    NEXT_TARGET   -> WRIST UP
    CONFIRM       -> MAKE A FIST
    ABORT         -> WRIST DOWN

Click Start to begin writing one continuous ``session.bin`` byte log.  Only
the HOLD phase is labelled for training.  Pausing keeps the serial port
draining and the raw log growing, but no paused frame is trainable; an active
trial interrupted by Pause is repeated after Resume.  The session ends
automatically after every planned trial has passed its contact/link checks and
a short unlabelled post-action verification phase.

The sidecar ``session.json`` stores frame-index label boundaries, device
timestamps, the randomized schedule, pause spans, parser statistics, and the
gesture map.  Full-rate raw bytes remain the source of truth.
"""

import argparse
import collections
import datetime
from dataclasses import dataclass
import json
import math
import pathlib
import secrets
import sys
import threading
import time

from emg_guided_session import (
    CLASSIFIER_PROTOCOL,
    COLLECTION_PROTOCOLS,
    EVENT_GATE_PROTOCOL,
    GESTURE_ACTIONS,
    GuidedSession,
    Phase,
    RUNNING_PHASES,
    StreamPosition,
    build_collection_plan,
)
from emg_protocol import TYPE_INFO, TYPE_RAW, PacketParser, decode_info, decode_raw
from emg_record import Recording, print_summary


EXPECTED_CHANNELS = 3
PLOT_COLORS = ("#2b8cbe", "#e34a33", "#31a354")
POLL_MS = 50
PLOT_PERIOD_SEC = 0.10
STREAM_STALE_SEC = 0.50
CONTACT_PREFLIGHT_SEC = 0.50
READ_SIZE = 1024


@dataclass(frozen=True)
class StreamSnapshot:
    """A copy-safe view of state owned by the serial worker thread."""

    info: object | None
    channels: tuple[tuple[int, ...], ...]
    attached: tuple[bool, ...]
    stream_age_sec: float
    accepted_packets: int
    lost_packets: int
    malformed_packets: int
    error: str
    recording: bool
    session_ready: bool
    position: StreamPosition

    @property
    def all_attached(self):
        return (
            len(self.attached) == EXPECTED_CHANNELS
            and all(self.attached)
        )

    @property
    def stream_fresh(self):
        return self.stream_age_sec <= STREAM_STALE_SEC


class SerialWorker:
    """Own serial reads and raw-log writes outside the GUI thread."""

    def __init__(self, connection, port, window_seconds):
        self.connection = connection
        self.port = port
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._closed = False

        self._preview_parser = PacketParser()
        self._info = None
        self._channels = []
        self._attached = ()
        self._last_raw_monotonic = None
        self._error = ""

        self._session_log = None
        self._session_path = None
        self._session_recording = None
        self._session_started_perf = None
        self._session_started_wall = None
        self._last_session_position = StreamPosition(0)
        self._pending_finish = None
        self._finished_result = None

    def start(self):
        self._thread.start()
        self._started = True

    def _run(self):
        try:
            while not self._stop.is_set():
                chunk = self.connection.read(READ_SIZE)
                if not chunk:
                    continue
                now = time.perf_counter()
                with self._lock:
                    if self._session_log is not None:
                        self._session_log.write(chunk)
                        self._session_recording.feed(chunk)
                    packets = self._preview_parser.feed(chunk)
                    for packet in packets:
                        self._absorb_preview_packet(packet, now)
        except Exception as error:  # Serial backends raise several OSError types.
            if not self._stop.is_set():
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
            self._stop.set()

    def _absorb_preview_packet(self, packet, now):
        if packet.type == TYPE_INFO:
            info = decode_info(packet.payload)
            if (
                self._info is not None
                and info.channel_count != self._info.channel_count
            ):
                raise RuntimeError("channel count changed while the stream was live")
            self._info = info
            if not self._channels:
                span = max(64, int(self.window_seconds * info.sample_rate_hz))
                self._channels = [
                    collections.deque([1 << (info.adc_bits - 1)] * span, maxlen=span)
                    for _ in range(info.channel_count)
                ]
                self._attached = tuple(False for _ in range(info.channel_count))
        elif packet.type == TYPE_RAW and self._info is not None:
            block = decode_raw(packet.payload, self._info.channel_count)
            for frame in block.frames:
                for index, value in enumerate(frame):
                    self._channels[index].append(int(value))
            self._attached = tuple(
                block.channel_attached(index)
                for index in range(self._info.channel_count)
            )
            self._last_raw_monotonic = now

    def start_recording(self, path):
        """Open an exclusive byte log; subsequent worker reads go into it."""
        path = pathlib.Path(path)
        with self._lock:
            if self._session_log is not None or self._pending_finish is not None:
                raise RuntimeError("a recording is already active")
            log = path.open("xb")
            self._session_log = log
            self._session_path = path
            self._session_recording = Recording()
            self._session_started_perf = time.perf_counter()
            self._session_started_wall = datetime.datetime.now().astimezone()
            self._last_session_position = StreamPosition(0)
            self._finished_result = None

    @staticmethod
    def _position_from_recording(recording, channel_hint=EXPECTED_CHANNELS):
        if recording is None:
            return StreamPosition(0)
        summary = recording.summary(0.0)
        channel_count = (
            recording.info.channel_count
            if recording.info is not None
            else channel_hint
        )
        detached = summary.get("frames_detached_by_channel", {})
        parser = summary["parser"]
        return StreamPosition(
            frame_index=recording.raw_frames,
            timestamp_us=recording.last_raw_timestamp_us,
            detached_by_channel=tuple(
                int(detached.get(str(index), 0))
                for index in range(channel_count)
            ),
            lost_packets=int(parser["lost"]),
            malformed_packets=int(parser["malformed"]),
            duplicated_packets=int(parser["duplicated"]),
            time_reversed_packets=int(parser["time_reversed"]),
        )

    def snapshot(self):
        now = time.perf_counter()
        with self._lock:
            stats = self._preview_parser.stats
            age = (
                float("inf")
                if self._last_raw_monotonic is None
                else max(0.0, now - self._last_raw_monotonic)
            )
            position = self._position_from_recording(self._session_recording)
            if self._session_recording is not None:
                self._last_session_position = position
            return StreamSnapshot(
                info=self._info,
                channels=tuple(tuple(channel) for channel in self._channels),
                attached=tuple(self._attached),
                stream_age_sec=age,
                accepted_packets=stats.accepted,
                lost_packets=stats.lost,
                malformed_packets=stats.malformed,
                error=self._error,
                recording=self._session_log is not None,
                session_ready=(
                    self._session_recording is not None
                    and self._session_recording.info is not None
                    and self._session_recording.raw_frames > 0
                ),
                position=(
                    position
                    if self._session_recording is not None
                    else self._last_session_position
                ),
            )

    def finish_recording(self):
        """Idempotently flush/close the byte log and return its summary.

        The active log is detached from the reader before file I/O.  If flush
        or close raises, the pending object remains available so the exact
        same finish operation can be retried without accepting more bytes.
        """
        with self._lock:
            if self._finished_result is not None:
                return self._finished_result
            if self._pending_finish is None:
                if self._session_log is None:
                    raise RuntimeError("no recording is active")
                elapsed = time.perf_counter() - self._session_started_perf
                summary = self._session_recording.summary(elapsed)
                summary["started"] = self._session_started_wall.isoformat()
                summary["port"] = self.port
                final_position = self._position_from_recording(
                    self._session_recording
                )
                self._pending_finish = (
                    self._session_log,
                    summary,
                    final_position,
                )
                self._session_log = None
                self._session_recording = None
                self._session_started_perf = None
                self._session_started_wall = None
            log, summary, final_position = self._pending_finish

        if not log.closed:
            log.flush()
            log.close()

        with self._lock:
            self._last_session_position = final_position
            self._finished_result = (summary, final_position)
            self._pending_finish = None
            return self._finished_result

    def close(self):
        if self._closed:
            return
        errors = []
        self._stop.set()
        cancel_read = getattr(self.connection, "cancel_read", None)
        if callable(cancel_read):
            try:
                cancel_read()
            except (OSError, RuntimeError):
                pass
        try:
            self.connection.close()
        except (OSError, RuntimeError) as error:
            errors.append(error)
        if self._started:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                errors.append(RuntimeError("serial worker did not stop"))
        with self._lock:
            recording_needs_finish = (
                self._session_log is not None or self._pending_finish is not None
            )
        if recording_needs_finish:
            try:
                self.finish_recording()
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self._closed = True


def _safe_session_id(value):
    if not value or pathlib.Path(value).name != value or value in (".", ".."):
        raise ValueError("session ID must be one non-empty path component")
    return value


def write_manifest(path, payload):
    """Atomically replace the sidecar after the full JSON is on disk."""
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CaptureApp:
    """Tk main-thread controller for one labelled collection session."""

    def __init__(self, root, worker, arguments, seed, tk, ttk, figure_class,
                 canvas_class):
        self.root = root
        self.worker = worker
        self.arguments = arguments
        self.seed = seed
        self.tk = tk
        self.ttk = ttk
        self.session = GuidedSession(
            build_collection_plan(
                arguments.protocol,
                arguments.repetitions,
                seed,
            ),
            protocol=arguments.protocol,
            prepare_seconds=arguments.prepare_seconds,
            transition_seconds=arguments.transition_seconds,
            active_seconds=arguments.active_seconds,
            verification_seconds=arguments.verification_seconds,
            recovery_seconds=arguments.recovery_seconds,
        )

        self.recording_started = False
        self.waiting_for_preflight = False
        self.preflight_ready_since = None
        self.finalized = False
        self._final_summary = None
        self._final_position = None
        self._final_status = None
        self._final_error = ""
        self._closing = False
        self._after_id = None
        self.session_id = None
        self.session_dir = None
        self.bin_path = None
        self.json_path = None
        self.last_plot_at = 0.0
        self.last_snapshot = self.worker.snapshot()

        self.prompt_var = tk.StringVar(value="CHECK SIGNAL")
        self.phase_var = tk.StringVar(value="Waiting for three-channel INFO/RAW")
        self.progress_var = tk.StringVar(
            value=f"0 / {self.session.total_trials} valid trials"
        )
        self.contact_var = tk.StringVar(value="ch0 --   ch1 --   ch2 --")
        self.result_var = tk.StringVar(value="Preview only; nothing is saved yet.")

        root.title(f"Objective 3.5 {arguments.protocol} EMG capture")
        root.geometry("1120x820")
        root.protocol("WM_DELETE_WINDOW", self.close)

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)
        ttk.Label(
            controls,
            textvariable=self.prompt_var,
            anchor="center",
            font=("Sans", 26, "bold"),
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            controls,
            textvariable=self.phase_var,
            anchor="center",
            font=("Sans", 14),
        ).pack(fill=tk.X)
        ttk.Label(
            controls,
            textvariable=self.progress_var,
            anchor="center",
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            controls,
            textvariable=self.contact_var,
            anchor="center",
        ).pack(fill=tk.X)

        figure = figure_class(figsize=(10.5, 5.4), dpi=100)
        self.axes = figure.subplots(EXPECTED_CHANNELS, 1, sharex=True)
        self.traces = []
        for index, axis in enumerate(self.axes):
            trace, = axis.plot([0], [0], color=PLOT_COLORS[index], linewidth=0.7)
            axis.grid(alpha=0.25)
            axis.set_ylabel(f"ch{index}\nraw-mid")
            self.traces.append(trace)
        self.axes[-1].set_xlabel(
            f"most recent {arguments.window_seconds:g} s; display centered only"
        )
        figure.tight_layout()
        self.canvas = canvas_class(figure, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

        button_row = ttk.Frame(root, padding=10)
        button_row.pack(fill=tk.X)
        self.start_button = ttk.Button(
            button_row,
            text="Start collection",
            command=self.start_collection,
            state=tk.DISABLED,
        )
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self.pause_button = ttk.Button(
            button_row,
            text="Pause",
            command=self.toggle_pause,
            state=tk.DISABLED,
        )
        self.pause_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self.stop_button = ttk.Button(
            button_row,
            text="Stop and save",
            command=self.stop_collection,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Label(root, textvariable=self.result_var, anchor="center").pack(
            fill=tk.X, padx=10, pady=(0, 10)
        )

        self._after_id = self.root.after(POLL_MS, self.poll)

    def _ready_for_start(self, snapshot):
        return (
            snapshot.info is not None
            and snapshot.info.channel_count == EXPECTED_CHANNELS
            and snapshot.all_attached
            and snapshot.stream_fresh
            and not snapshot.error
        )

    def start_collection(self):
        if self.recording_started or self.finalized:
            return
        snapshot = self.worker.snapshot()
        if not self._ready_for_start(snapshot):
            self.result_var.set("Fix stream/contact first; Start remains blocked.")
            return

        created_session_dir = False
        try:
            session_id = self.arguments.session_id
            if session_id is None:
                session_id = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.session_id = _safe_session_id(session_id)
            root = pathlib.Path(self.arguments.out_root)
            root.mkdir(parents=True, exist_ok=True)
            self.session_dir = root / self.session_id
            self.session_dir.mkdir(exist_ok=False)
            created_session_dir = True
            self.bin_path = self.session_dir / "session.bin"
            self.json_path = self.session_dir / "session.json"
            self.worker.start_recording(self.bin_path)
        except (OSError, RuntimeError, ValueError) as error:
            if created_session_dir:
                try:
                    self.session_dir.rmdir()
                except OSError:
                    pass
            self.result_var.set(f"Could not start: {error}")
            return

        self.recording_started = True
        self.waiting_for_preflight = True
        self.preflight_ready_since = None
        self.start_button.configure(state=self.tk.DISABLED)
        self.stop_button.configure(state=self.tk.NORMAL)
        self.prompt_var.set("PREFLIGHT")
        self.phase_var.set("Recording started; waiting for fresh INFO and contact")
        self.result_var.set(f"Writing raw bytes to {self.bin_path}")

    def toggle_pause(self):
        if not self.recording_started or self.finalized:
            return
        snapshot = self.worker.snapshot()
        now = time.perf_counter()
        if self.session.phase is Phase.PAUSED:
            if not snapshot.all_attached or not snapshot.stream_fresh:
                self.result_var.set(
                    "Cannot resume until all channels are fresh/attached."
                )
                return
            self.session.resume(now, snapshot.position)
            self.pause_button.configure(text="Pause")
        elif self.session.phase in RUNNING_PHASES:
            self.session.pause(now, snapshot.position, "manual_pause")
            self.pause_button.configure(text="Resume")

    def stop_collection(self):
        if self.finalized:
            self.close()
            return
        if self._final_summary is not None:
            self.finalize(self._final_status, self._final_error)
            return
        if not self.recording_started:
            self.close()
            return
        snapshot = self.worker.snapshot()
        self.session.stop(time.perf_counter(), snapshot.position, "manual_stop")
        self.finalize("stopped")

    def _auto_pause(self, reason, snapshot):
        if self.session.phase in RUNNING_PHASES:
            self.session.pause(time.perf_counter(), snapshot.position, reason)
            self.pause_button.configure(text="Resume")

    def _update_protocol(self, snapshot, now):
        if self.waiting_for_preflight:
            ready = (
                snapshot.session_ready
                and snapshot.all_attached
                and snapshot.stream_fresh
            )
            if ready:
                if self.preflight_ready_since is None:
                    self.preflight_ready_since = now
                elif now - self.preflight_ready_since >= CONTACT_PREFLIGHT_SEC:
                    self.session.set_sample_rate_hz(
                        int(snapshot.info.sample_rate_hz)
                    )
                    self.session.start(now)
                    self.waiting_for_preflight = False
                    self.pause_button.configure(state=self.tk.NORMAL)
            else:
                self.preflight_ready_since = None
            return

        if self.session.phase in RUNNING_PHASES:
            if not snapshot.stream_fresh:
                self._auto_pause("stream_stale", snapshot)
            elif not snapshot.all_attached:
                self._auto_pause("contact_lost", snapshot)
            else:
                self.session.advance(now, snapshot.position)

        if self.session.phase is Phase.COMPLETE and not self.finalized:
            self.finalize("complete")

    def _phase_text(self, now):
        trial = self.session.current_trial
        remaining = self.session.remaining_seconds(now)
        if self.waiting_for_preflight:
            return "PREFLIGHT", "Waiting for stable contact and session INFO"
        if self.session.phase is Phase.PREPARE:
            if self.session.protocol == EVENT_GATE_PROTOCOL:
                return (
                    "RELAX",
                    f"Remain neutral; next {trial.action} · {remaining:0.1f} s",
                )
            return (
                f"NEXT: {trial.action}",
                f"Prepare {trial.label} · {remaining:0.1f} s",
            )
        if self.session.phase is Phase.TRANSITION:
            if trial.label == "REST":
                return (
                    "STAY RELAXED",
                    f"REST transition is not labelled · {remaining:0.1f} s",
                )
            return (
                f"MOVE NOW: {trial.action}",
                f"Transition is not labelled · {remaining:0.1f} s",
            )
        if self.session.phase is Phase.ACTIVE:
            return (
                f"HOLD: {trial.action}",
                f"Recording label {trial.label} · {remaining:0.1f} s",
            )
        if self.session.phase is Phase.VERIFY:
            return (
                "RELAX",
                f"Checking packet quality; not labelled · {remaining:0.1f} s",
            )
        if self.session.phase is Phase.RECOVERY:
            return "RELAX", f"Recovery is not labelled · {remaining:0.1f} s"
        if self.session.phase is Phase.PAUSED:
            return "PAUSED — RELAX", self.session.last_result
        if self.session.phase is Phase.COMPLETE:
            return "SESSION COMPLETE", "All valid trials were saved"
        if self.session.phase is Phase.STOPPED:
            return "SESSION STOPPED", "Completed trials were preserved"
        return "CHECK SIGNAL", "Preview only; click Start when all contacts are green"

    def _update_plot(self, snapshot):
        if len(snapshot.channels) != EXPECTED_CHANNELS:
            return
        midpoint = (
            1 << (snapshot.info.adc_bits - 1)
            if snapshot.info is not None
            else 2048
        )
        for index, (axis, trace, values) in enumerate(
            zip(self.axes, self.traces, snapshot.channels)
        ):
            centered = [value - midpoint for value in values]
            positions = range(len(centered))
            trace.set_data(positions, centered)
            limit = max(64, int(1.15 * max(abs(min(centered)), abs(max(centered)))))
            axis.set_xlim(0, max(1, len(centered) - 1))
            axis.set_ylim(-limit, limit)
            contact = (
                "attached"
                if index < len(snapshot.attached) and snapshot.attached[index]
                else "NO CONTACT"
            )
            rail_hits = sum(value <= 0 or value >= 4095 for value in values)
            warning = f" · CLIPPING x{rail_hits}" if rail_hits else ""
            axis.set_title(
                f"ch{index} · raw ADC · {contact}{warning}",
                loc="left",
                fontsize=9,
                color=(
                    "#b2182b"
                    if contact == "NO CONTACT" or rail_hits
                    else "black"
                ),
            )
        self.canvas.draw_idle()

    def poll(self):
        snapshot = self.worker.snapshot()
        self.last_snapshot = snapshot
        now = time.perf_counter()

        if snapshot.error and self.recording_started and not self.finalized:
            self.session.stop(now, snapshot.position, "serial_error")
            self.finalize("error", extra_error=snapshot.error)
        elif self.recording_started and not self.finalized:
            self._update_protocol(snapshot, now)

        if now - self.last_plot_at >= PLOT_PERIOD_SEC:
            self._update_plot(snapshot)
            self.last_plot_at = now

        prompt, phase = self._phase_text(now)
        self.prompt_var.set(prompt)
        self.phase_var.set(phase)
        self.progress_var.set(
            f"{self.session.completed_trials} / {self.session.total_trials} "
            "valid trials"
        )
        contacts = []
        for index in range(EXPECTED_CHANNELS):
            attached = index < len(snapshot.attached) and snapshot.attached[index]
            contacts.append(f"ch{index} {'OK' if attached else 'NO CONTACT'}")
        age = snapshot.stream_age_sec
        age_text = "--" if not math_is_finite(age) else f"{age * 1000:.0f} ms"
        self.contact_var.set(
            "   ".join(contacts)
            + f"   stream age {age_text}   lost {snapshot.lost_packets}"
        )

        if (
            not self.recording_started
            and not self.finalized
            and self._final_summary is None
        ):
            state = (
                self.tk.NORMAL
                if self._ready_for_start(snapshot)
                else self.tk.DISABLED
            )
            self.start_button.configure(state=state)

        if not self._closing:
            self._after_id = self.root.after(POLL_MS, self.poll)

    def _save_session(self, status, extra_error=""):
        """Finish raw bytes and write JSON without depending on live Tk state."""
        if self.finalized:
            return self._final_summary
        if self._final_status is None:
            self._final_status = status
            self._final_error = extra_error
        if self._final_summary is None:
            summary, final_position = self.worker.finish_recording()
            self._final_summary = summary
            self._final_position = final_position
            self.recording_started = False
            self.waiting_for_preflight = False
        else:
            summary = self._final_summary
            final_position = self._final_position
        manifest = self.session.to_manifest(
            seed=self.seed,
            status=self._final_status,
        )
        manifest.update(
            {
                "session_id": self.session_id,
                "created": summary.get("started"),
                "port": self.arguments.port,
                "raw_log": self.bin_path.name,
                "final_stream_position": final_position.to_dict(),
                "stream_summary": summary,
            }
        )
        if self._final_error:
            manifest["error"] = self._final_error
        write_manifest(self.json_path, manifest)
        self.finalized = True
        return summary

    def emergency_finalize(self, reason="mainloop_exit"):
        """Best-effort save for exceptions or termination outside Tk callbacks."""
        if self.finalized:
            return True
        if not self.recording_started and self._final_summary is None:
            return True
        try:
            if self.recording_started:
                snapshot = self.worker.snapshot()
                self.session.stop(
                    time.perf_counter(),
                    snapshot.position,
                    reason,
                )
            summary = self._save_session("stopped", reason)
        except Exception as error:
            print(f"Emergency session save failed: {error}", file=sys.stderr)
            return False
        print_summary(summary)
        print(f"  emergency save wrote {self.bin_path} and {self.json_path}")
        return True

    def finalize(self, status, extra_error=""):
        if self.finalized:
            return True
        try:
            summary = self._save_session(status, extra_error)
        except (OSError, RuntimeError, ValueError) as error:
            self.result_var.set(f"Could not finalize session: {error}")
            self.pause_button.configure(state=self.tk.DISABLED)
            self.stop_button.configure(text="Retry save", state=self.tk.NORMAL)
            return False

        self.waiting_for_preflight = False
        self.pause_button.configure(state=self.tk.DISABLED)
        self.start_button.configure(state=self.tk.DISABLED)
        self.stop_button.configure(text="Close", state=self.tk.NORMAL)
        print_summary(summary)
        print(f"  wrote {self.bin_path} and {self.json_path}")
        self.result_var.set(
            f"Saved: {self.session_dir} ({self._final_status})"
        )
        return True

    def close(self):
        if self._closing:
            return
        if self.recording_started and not self.finalized:
            snapshot = self.worker.snapshot()
            self.session.stop(
                time.perf_counter(),
                snapshot.position,
                "window_closed",
            )
            if not self.finalize("stopped"):
                return
        elif self._final_summary is not None and not self.finalized:
            if not self.finalize(self._final_status, self._final_error):
                return
        self._closing = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except self.tk.TclError:
                pass
            self._after_id = None
        try:
            self.worker.close()
        finally:
            self.root.destroy()


def math_is_finite(value):
    """Avoid importing NumPy just to format one age value."""
    return math.isfinite(value)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=COLLECTION_PROTOCOLS,
        default=CLASSIFIER_PROTOCOL,
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--out-root")
    parser.add_argument("--session-id")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--prepare-seconds", type=float)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument("--active-seconds", type=float)
    parser.add_argument("--verification-seconds", type=float, default=0.1)
    parser.add_argument("--recovery-seconds", type=float)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    event_gate = arguments.protocol == EVENT_GATE_PROTOCOL
    if arguments.out_root is None:
        arguments.out_root = (
            "datasets/emg_event_gate" if event_gate else "datasets/emg"
        )
    if arguments.repetitions is None:
        arguments.repetitions = 3 if event_gate else 5
    if arguments.prepare_seconds is None:
        arguments.prepare_seconds = 0.5 if event_gate else 2.0
    if arguments.active_seconds is None:
        arguments.active_seconds = 2.0 if event_gate else 3.0
    if arguments.recovery_seconds is None:
        arguments.recovery_seconds = 0.5 if event_gate else 1.5
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    seed = arguments.seed if arguments.seed is not None else secrets.randbits(32)

    try:
        plan = build_collection_plan(
            arguments.protocol,
            arguments.repetitions,
            seed,
        )
        GuidedSession(
            plan,
            protocol=arguments.protocol,
            prepare_seconds=arguments.prepare_seconds,
            transition_seconds=arguments.transition_seconds,
            active_seconds=arguments.active_seconds,
            verification_seconds=arguments.verification_seconds,
            recovery_seconds=arguments.recovery_seconds,
        )
        if not math.isfinite(arguments.window_seconds) or arguments.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        if arguments.session_id is not None:
            _safe_session_id(arguments.session_id)
    except (TypeError, ValueError) as error:
        print(f"Invalid configuration: {error}")
        return 2

    try:
        import serial
    except ImportError:
        print("pyserial is required: install the same dependency used by emg_scope.py")
        return 2

    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except Exception as error:
        print(f"Could not import the GUI dependencies: {error}")
        return 1

    connection = None
    worker = None
    root = None
    app = None
    exit_reason = "mainloop_exit"
    try:
        connection = serial.Serial(arguments.port, 115200, timeout=0.02)
        connection.reset_input_buffer()
        root = tk.Tk()
        worker = SerialWorker(
            connection,
            arguments.port,
            arguments.window_seconds,
        )
        worker.start()
        app = CaptureApp(
            root,
            worker,
            arguments,
            seed,
            tk,
            ttk,
            Figure,
            FigureCanvasTkAgg,
        )
        print(f"Randomization seed: {seed}")
        print(f"Collection protocol: {arguments.protocol}")
        print("Click Start in the GUI after all three contact indicators are OK.")
        root.mainloop()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("Capture interrupted; preserving the partial session.")
        return 130
    except (OSError, serial.SerialException) as error:
        exit_reason = "serial_error"
        print(f"Could not open or read {arguments.port}: {error}")
        print(
            "Activate the dialout group in a normal user session; "
            "do not run the GUI as root."
        )
        return 1
    except Exception as error:
        exit_reason = f"gui_error: {type(error).__name__}: {error}"
        print(f"Could not create or run the GUI: {error}")
        return 1
    finally:
        if app is not None:
            app.emergency_finalize(exit_reason)
        try:
            if worker is not None:
                worker.close()
            elif connection is not None:
                connection.close()
        except Exception as error:
            print(f"Resource cleanup failed: {error}", file=sys.stderr)
        finally:
            if app is not None and not app.finalized:
                app.emergency_finalize(exit_reason)
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
