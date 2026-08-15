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
 * The threshold combines a relative rule with an absolute floor:
 * max(factor x rest baseline, floor). Each rule covers the other's measured
 * failure. An absolute count alone breaks across louder donnings (donning B
 * rested at twice donning A); a rest multiple alone breaks when contact
 * improves, because rest amplitude is a contact-noise figure while the
 * preparation/gesture band is set by physiology and electrode placement —
 * re-gelling one noisy electrode dropped rest five-fold and left that band
 * nearly still. The baseline is an integer EMA over windows the classifier
 * itself called REST. Suppressed windows do not feed it, or a long
 * sub-threshold movement would drag the threshold up under the user.
 *
 * Deliberate choices, written down because each one is arguable:
 *  - Until the first classified-REST window seeds the baseline, only the
 *    floor judges. Fail-open above it on purpose: a forceful cold-start
 *    ABORT stays reachable and ordinary events cannot fire before the gate
 *    has observed rest anyway, while sub-floor cold-start movement is
 *    suppressed instead of passing unjudged as it originally did.
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

/* Frozen 2026-08-14 by an independent self-paced session that took no part in
 * choosing them: `datasets/emg_activation/selfpaced_20260814_143300.bin`, six
 * gestures, all six emitted correctly with no spurious or missing event. That
 * recording also suppressed 88 windows across 17 low-amplitude episodes, three
 * of which ran 21, 17, and 11 windows — the first two longer than the twelve
 * hold-off windows plus a five-window run, so each would have emitted a
 * spurious event on the previous firmware, and both were ABORT-dominated.
 * Counts are in 50 ms feature hops; K multiplies the rest baseline.
 *
 * How they were chosen. A joint sweep over the defect recording
 * (`selfpaced_20260814_defect.bin`) and the frozen 9/9 event-gate session
 * passed the whole band K = 2..5; K >= 6 starts suppressing real gestures and
 * missing events. Events did not separate the survivors, so the mechanism
 * did: at K = 3 the entire preparatory movement (peak 79 against a 35
 * baseline) falls below the 105 threshold, while at K = 2 four of its windows
 * cross a 70 threshold and are stopped only because they fragment into runs of
 * two. Passing by fragmentation is not margin. The acceptance session then
 * settled it independently — its tightest suppressed episode peaked at 90
 * against a 93 threshold and ran 21 windows, so K = 2 would have let it
 * through and fired a false ABORT.
 *
 * A run-length margin was tried as a selector and discarded: it jumps
 * non-monotonically (6, 7, 29, 30 across shifts at fixed K) because one
 * marginal window splits a run in half, so it measures threshold proximity
 * rather than robustness.
 *
 * Shift 4 gives the EMA most of a step change in ~16 hops = 0.8 s. No
 * recording so far discriminates between shifts, so it remains the middle of
 * the swept range rather than a measured value — weaker evidence than K has.
 *
 * The margin is thin and should be re-measured. Three counts separated the
 * tightest episode from its threshold; that is not yet enough to tell a
 * robust K = 3 from a second lucky one. */
#define EMG_ACTIVATION_FACTOR 3u
#define EMG_ACTIVATION_BASELINE_SHIFT 4u

/* Interim absolute floor under the relative rule, measured 2026-08-15 and
 * applied as threshold = max(factor x baseline, floor). Not frozen: two
 * donnings of evidence and no independent acceptance session yet.
 *
 * Why the relative rule alone collapsed: re-gelling a noisy ch0 electrode
 * dropped resting total MAV from 32 to a drifting 6-43, so 3 x baseline
 * swung between 18 and 129. The band the threshold must land in barely
 * moved: preparatory movement measured 73 (65-79 the day before) and the
 * weakest deliberate gesture 145 (318 the day before). Rest 6 would need
 * K = 18 while rest 43 allows at most K = 3 — no single factor spans a
 * seven-fold noise drift across an almost-still gesture band.
 *
 * 110 sits 31 counts above the loudest measured preparation (79) and 35
 * below the weakest measured gesture (145), and stays compatible with the
 * frozen acceptance session: its suppressed episodes peaked at 90 and its
 * gestures ran 318-736, so the floor changes no decision that mattered
 * there. It also closes the cold-start gap — sub-floor movement before the
 * first classified-REST window is now suppressed rather than passed.
 *
 * The real fix is the roadmapped per-donning calibration (rest plus
 * comfortable effort), which replaces these constants with measured ones;
 * this floor then remains the uncalibrated fail-safe default. */
#define EMG_ACTIVATION_THRESHOLD_FLOOR 110

/* Sums of three per-channel window MAVs are bounded by construction; the
 * clamp only defends the accumulator against a corrupt caller. */
#define EMG_ACTIVATION_TOTAL_LIMIT (3 * 32767)

typedef struct {
    int32_t accumulator; /* baseline << baseline_shift; EMA state */
    int32_t threshold_floor;
    uint16_t factor;
    uint16_t baseline_shift;
    bool has_baseline;
} emg_activation_t;

/* Rejects factor 0, which would keep the code path alive while judging
 * nothing; shifts outside 1..8 — 0 makes the baseline one window's value,
 * which a single transition window can spike, and above 8 the seed shift
 * could overflow the accumulator; and floors outside
 * 1..EMG_ACTIVATION_TOTAL_LIMIT-1 — 0 removes the collapse and cold-start
 * protection, and at or above the clamp limit nothing could ever pass. */
bool emg_activation_init(emg_activation_t *activation, uint16_t factor,
                         uint16_t baseline_shift, int32_t threshold_floor);

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
