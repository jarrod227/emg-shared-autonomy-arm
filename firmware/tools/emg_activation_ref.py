"""Bit-exact Python mirror of firmware/src/emg_activation.c.

The activation threshold rewrites low-activation non-REST classifier
decisions to REST before they reach the event gate, closing the measured
preparatory-movement defect: a gesture-shaped window at preparation
amplitude is not an intent.

This mirror exists so offline sweeps and firmware-versus-host replays judge
windows with the very arithmetic the MCU uses — integer EMA, integer
threshold, same clamps, same order of operations. Divergence between the two
is caught by test_emg_activation_ref.py against the fixture that
test_emg_activation.c emits, decision for decision and baseline for
baseline.

The decision is threshold = max(factor x rest baseline, floor). The relative
rule alone collapsed on 2026-08-15 when re-gelling a noisy electrode dropped
rest from 32 to a drifting 6-43 while the preparation/gesture band barely
moved (preparation 73, weakest gesture 145): rest amplitude is contact
noise, the band is physiology, and the two do not covary.

FROZEN_FACTOR and FROZEN_BASELINE_SHIFT were frozen 2026-08-14 by an
independent self-paced session that took no part in choosing them.
THRESHOLD_FLOOR is interim, measured from two donnings on 2026-08-15 and
not yet independently accepted; per-donning calibration is the planned
replacement, with the floor remaining as the uncalibrated default. See
emg_activation.h for the full reasoning and the recording names. Kept in
step with EMG_ACTIVATION_FACTOR / EMG_ACTIVATION_BASELINE_SHIFT /
EMG_ACTIVATION_THRESHOLD_FLOOR there.
"""

REST = "REST"

FROZEN_FACTOR = 3
FROZEN_BASELINE_SHIFT = 4
THRESHOLD_FLOOR = 110

# Sums of three per-channel window MAVs are bounded by construction; the
# clamp only defends the accumulator against a corrupt caller.
TOTAL_LIMIT = 3 * 32767


class ActivationGate:
    """Stateful mirror of emg_activation_t plus its two operations."""

    def __init__(self, factor=FROZEN_FACTOR,
                 baseline_shift=FROZEN_BASELINE_SHIFT,
                 threshold_floor=THRESHOLD_FLOOR):
        factor = int(factor)
        baseline_shift = int(baseline_shift)
        threshold_floor = int(threshold_floor)
        if factor <= 0:
            raise ValueError("factor 0 would judge nothing")
        if not 1 <= baseline_shift <= 8:
            raise ValueError("baseline_shift must be in 1..8")
        if not 1 <= threshold_floor < TOTAL_LIMIT:
            raise ValueError(
                "threshold_floor must be in 1..TOTAL_LIMIT-1"
            )
        self.factor = factor
        self.baseline_shift = baseline_shift
        self.threshold_floor = threshold_floor
        self.accumulator = 0
        self.has_baseline = False

    @property
    def baseline(self):
        """Baseline in MAV counts; 0 until the first classified-REST window."""
        if not self.has_baseline:
            return 0
        return self.accumulator >> self.baseline_shift

    def apply(self, prediction, *, valid=True, total_mav=0):
        """Return the decision the gate should see.

        Either the prediction itself, or REST when the shape arrived without
        the activation of an intent. Invalid windows pass through unchanged
        and update nothing, matching the C side.
        """
        if not valid:
            return prediction
        total = int(total_mav)
        if total < 0:
            total = 0
        elif total > TOTAL_LIMIT:
            total = TOTAL_LIMIT

        if prediction == REST:
            if not self.has_baseline:
                self.accumulator = total << self.baseline_shift
                self.has_baseline = True
            else:
                # acc += A - (acc >> s) keeps the sub-LSB residue inside the
                # accumulator, so a small persistent drift moves the baseline
                # instead of dying in a deadband.
                self.accumulator += total - (
                    self.accumulator >> self.baseline_shift
                )
            return REST

        if total < self.threshold_floor:
            # The floor judges before and after the baseline exists: the
            # preparation/gesture band is physiology, not contact noise,
            # so it needs no rest observation to apply.
            return REST
        if not self.has_baseline:
            # No rest observed yet and the floor is cleared: fail open so
            # a forceful cold-start ABORT stays reachable; ordinary events
            # cannot fire before the gate has seen rest anyway.
            return prediction
        if total < self.baseline * self.factor:
            return REST
        return prediction
