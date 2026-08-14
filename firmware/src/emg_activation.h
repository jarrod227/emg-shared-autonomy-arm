/* Rest-relative activation threshold in front of the event gate.
 *
 * The gate decides which gesture a window's shape resembles; nothing in it
 * asks how strongly the muscle is working. On 2026-08-14 that gap fired a
 * real event: a 0.85 s preparatory wrist extension before a fist was
 * classified correctly as extension, outlasted the onset hold-off, and
 * emitted NEXT_TARGET 0.15 s before the intended gesture. Duration cannot
 * close the gap — the preparation and the gesture were one continuous
 * non-REST run with the class changing mid-run, so there is no boundary for
 * a length rule to split on. Amplitude can: that recording put rest at ~29
 * total MAV, the spurious movement at 60-71, and intended gestures at
 * 318-736.
 *
 * This stage rewrites low-activation non-REST decisions to REST before they
 * reach the gate. Rewriting rather than dropping matters: REST is what lets
 * the gate re-arm, and "shape without activation" is exactly what rest means
 * here. A useful side effect is that the gate's onset hold-off then anchors
 * at the loud onset instead of at the quiet preparation, which is where the
 * classifier's onset confusion actually lives.
 *
 * The threshold is a multiple of a rest baseline, never an absolute count:
 * donning B ran at twice donning A's amplitude, so an absolute floor tuned
 * on one donning is useless or crippling on the next. The baseline is an
 * integer EMA over windows the classifier itself called REST. Suppressed
 * windows do not feed it, or a long sub-threshold movement would drag the
 * threshold up under the user.
 *
 * Deliberate choices, written down because each one is arguable:
 *  - Until the first classified-REST window seeds the baseline, everything
 *    passes through unjudged. Fail-open on purpose: it keeps a cold-start
 *    ABORT reachable, and ordinary events cannot fire before the gate has
 *    observed rest anyway.
 *  - The threshold applies to every class, ABORT included. The measured
 *    ABORT trials ran 15-22x rest against a 5x threshold, so the margin is
 *    real; a user whose stop gesture is weaker than that needs the factor
 *    re-measured, not a one-class exemption that reopens the defect.
 *  - The baseline survives stream discontinuities: rest amplitude is a
 *    property of the wearer and the donning, not of stream continuity.
 *  - A transition window that classifies as REST with elevated amplitude
 *    nudges (or, as the seed, inflates) the baseline. Known limitation;
 *    check it during the factor sweep rather than complicating the seed
 *    rule now.
 *
 * Pure logic with no HAL dependency, compiled into the firmware and into
 * the host tests, and mirrored bit-exactly by
 * firmware/tools/emg_activation_ref.py.
 */

#ifndef EMG_ACTIVATION_H
#define EMG_ACTIVATION_H

#include "emg_packet.h"

#include <stdbool.h>
#include <stdint.h>

/* CANDIDATE values, not validated. A joint offline sweep over the recording
 * that exposed the defect and the frozen 9/9 event-gate session put the
 * whole band K = 2..5 through both; K >= 6 starts suppressing real gestures
 * and missing events. Within that band the events do not separate the
 * choices, so the mechanism does: at K = 3 the entire preparatory movement
 * (peak 79 counts against a 35 baseline) falls below the 105 threshold,
 * while at K = 2 four of its windows cross a 70 threshold and are stopped
 * only because they fragment into runs of two, short of the gate's five.
 * Passing by fragmentation is not margin. K = 3 also sits two steps below
 * where real gestures start breaking.
 *
 * The run-length margin was tried as a selector and discarded: it jumps
 * non-monotonically (6, 7, 29, 30 across shifts at fixed K) because one
 * marginal window splits a run in half, so it measures threshold proximity
 * rather than robustness.
 *
 * Shift 4 makes the EMA cover most of a step change in ~16 hops = 0.8 s at
 * the 50 ms hop. The two recordings do not discriminate between shifts, so
 * this stays the middle of the swept range rather than a measured value.
 *
 * Both numbers came from two recordings, one of which was used to find the
 * defect and the other already spent as an acceptance set. The same rule
 * applies as to the gate counts: freeze only after a session that did not
 * participate in choosing them agrees, collected self-paced, since a cued
 * protocol cannot produce the preparatory movement at all. */
#define EMG_ACTIVATION_FACTOR 3u
#define EMG_ACTIVATION_BASELINE_SHIFT 4u

/* Sums of three per-channel window MAVs are bounded by construction; the
 * clamp only defends the accumulator against a corrupt caller. */
#define EMG_ACTIVATION_TOTAL_LIMIT (3 * 32767)

typedef struct {
    int32_t accumulator; /* baseline << baseline_shift; EMA state */
    uint16_t factor;
    uint16_t baseline_shift;
    bool has_baseline;
} emg_activation_t;

/* Rejects factor 0, which would keep the code path alive while judging
 * nothing, and shifts outside 1..8 — 0 makes the baseline one window's
 * value, which a single transition window can spike, and above 8 the seed
 * shift could overflow the accumulator. */
bool emg_activation_init(emg_activation_t *activation, uint16_t factor,
                         uint16_t baseline_shift);

/* Baseline in MAV counts; 0 until the first classified-REST window. */
int32_t emg_activation_baseline(const emg_activation_t *activation);

/* Feed one classifier decision plus the summed per-channel MAV of its
 * window. Returns the decision the gate should see: the prediction itself,
 * or REST when the shape arrived without the activation of an intent.
 * `valid` mirrors the gate's flag; invalid windows pass through and change
 * nothing here, because the gate discards all evidence on them anyway and a
 * second behaviour would only be a place for the two to disagree. */
emg_command_t emg_activation_apply(emg_activation_t *activation,
                                   emg_command_t prediction, bool valid,
                                   int32_t total_mav);

#endif /* EMG_ACTIVATION_H */
