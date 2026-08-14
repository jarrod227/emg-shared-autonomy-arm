"""Bit-exact Python mirror of firmware/src/emg_activation.c.

The rest-relative activation threshold rewrites low-activation non-REST
classifier decisions to REST before they reach the event gate, closing the
measured preparatory-movement defect: a gesture-shaped window at twice
resting amplitude is not an intent when intended gestures run ten to
twenty-five times rest.

This mirror exists so offline sweeps and firmware-versus-host replays judge
windows with the very arithmetic the MCU uses — integer EMA, integer
threshold, same clamps, same order of operations. Divergence between the two
is caught by test_emg_activation_ref.py against the fixture that
test_emg_activation.c emits, decision for decision and baseline for
baseline.

CANDIDATE_FACTOR and CANDIDATE_BASELINE_SHIFT are candidates, not validated
values, and must stay in step with emg_activation.h -- that header carries
the reasoning for the pair. In short: a joint sweep passed K = 2..5 on both
the defect recording and the frozen 9/9 session, and K = 3 was taken because
it clears the preparatory movement outright rather than by fragmenting it.
"""

REST = "REST"

CANDIDATE_FACTOR = 3
CANDIDATE_BASELINE_SHIFT = 4

# Sums of three per-channel window MAVs are bounded by construction; the
# clamp only defends the accumulator against a corrupt caller.
TOTAL_LIMIT = 3 * 32767


class ActivationGate:
    """Stateful mirror of emg_activation_t plus its two operations."""

    def __init__(self, factor=CANDIDATE_FACTOR,
                 baseline_shift=CANDIDATE_BASELINE_SHIFT):
        factor = int(factor)
        baseline_shift = int(baseline_shift)
        if factor <= 0:
            raise ValueError("factor 0 would judge nothing")
        if not 1 <= baseline_shift <= 8:
            raise ValueError("baseline_shift must be in 1..8")
        self.factor = factor
        self.baseline_shift = baseline_shift
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

        if not self.has_baseline:
            # No rest observed yet: fail open so a cold-start ABORT stays
            # reachable; ordinary events cannot fire before the gate has
            # seen rest anyway.
            return prediction
        if total < self.baseline * self.factor:
            return REST
        return prediction
