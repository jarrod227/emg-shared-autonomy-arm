/* Continuous view-control output, alongside the discrete event.
 *
 * The INTENT packet has always carried `direction` and `activation` next to
 * `command`, and main.c has always written 0 to both. They are the
 * proportional half of the same decision: `command` is the event gate's
 * output and fires at most once per gesture, while these two are a stream at
 * one per feature hop, for a consumer that is steering rather than
 * commanding.
 *
 * Both are derived from the *post-activation* decision, not the raw
 * classifier output. A window the activation stage rewrote to REST is one
 * where the muscle was not working hard enough to mean anything, and that is
 * exactly the window a view command must not act on either.
 *
 * The reference level was the open problem and is now a measured input.
 * Activation is a fraction of the span between the threshold and whatever the
 * wearer can actually produce for the driving gesture, and that ceiling is
 * measured per donning and per direction by emg_calibrate.py and arrives over
 * SET_ACTIVATION.
 *
 * It is per direction because one number provably cannot serve both. Measured
 * 2026-08-23 in a single capture on a single donning: wrist extension came to
 * 4.19x the session threshold and ulnar deviation to 5.51x. A constant cannot
 * span that, and the constant that used to try was 3, below both -- which
 * saturated 58% of one session's LEFT commands at full deflection, where they
 * carry no proportional information at all.
 *
 * Activation is zero when direction is zero, and that is a definition rather
 * than a fallback: activation means "this fraction of the span for the
 * gesture being commanded", so with no gesture there is no span to take a
 * fraction of. Publishing a number derived from whichever reference happened
 * to be picked would make the units depend on an arbitrary choice, and a
 * consumer already ignores activation when direction is HOLD.
 *
 * It must be an *instantaneous* ceiling, because that is what activation is
 * computed from. The first version of this constant was picked against
 * sustained_level -- the level held across seventeen windows, which is the
 * minimum of a run and therefore far lower -- and the board saturated at full
 * deflection through most of an ordinary hold. Measured 2026-08-20 on one
 * donning at threshold 68: the 90th percentile of instantaneous total MAV
 * during held gestures was 216 for wrist extension and 205 for ulnar
 * deviation, ratios of 3.2 and 3.0, against a sustained ceiling of 184 and
 * 193 and a peak of 338 and 341. The 90th percentile is the right statistic:
 * the median saturates half the time, and the peak puts full deflection out
 * of reach and is set by a single window.
 *
 * Radial deviation gives 1.8 on the same donning, which is the measurement
 * saying plainly that one constant cannot serve both directions -- it would
 * sit against the floor for one of them. That, plus reference/threshold
 * running 2.38, 3.34 and 2.47 across three donnings, is why this is interim.
 * It exists so a controllability experiment can happen before the protocol
 * grows a field to carry a measured reference. It sets the gain, not whether
 * the signal is steady enough to steer with, and it is the latter that is
 * unknown: every proportional measurement so far has been open loop, and the
 * held activation wandered 34% to 42% of its own mean with nothing to correct
 * against.
 */

#ifndef EMG_VIEW_H
#define EMG_VIEW_H

#include "emg_packet.h"

#include <stdint.h>

/* Fallback for an uncalibrated board: reference = threshold * NUM / DEN.
 * Known to be too low on every donning measured so far, so it is a way to
 * keep steering roughly rather than a default worth shipping into. */
#define EMG_VIEW_REFERENCE_NUM 3
#define EMG_VIEW_REFERENCE_DEN 1

/* -1, 0 or +1, per the INTENT payload. Only the gestures assigned to the two
 * view directions produce a non-zero value; every other decision, REST
 * included, is 0 and reads downstream as HOLD.
 *
 * NEXT_TARGET is the only direction wired today: the deployed model has four
 * classes and the second direction gesture is not one of them. */
int8_t emg_view_direction(emg_command_t decision);

/* Normalized activation, 0..65535 mapping to 0.0..1.0 as the protocol states.
 *
 * Zero at or below the threshold rather than negative: below it there is no
 * intent to scale, and the controller's own deadband then reads it as HOLD.
 * Saturates at the reference instead of extrapolating, so a wearer who pushes
 * past their calibrated ceiling gets full deflection rather than a wrapped
 * value. */
/* The calibrated ceiling for one direction, or the compile-time fallback
 * when that direction has none. Separated from emg_view_activation so the
 * fallback rule has one home and can be tested without a whole packet. */
int32_t emg_view_reference(int8_t direction, int32_t threshold,
                           int32_t reference_left, int32_t reference_right);

/* Zero when direction is 0, when the reference does not exceed the
 * threshold (an empty span), or when the window is below the threshold. */
uint16_t emg_view_activation(int32_t total_mav, int32_t threshold,
                             int32_t reference);

#endif /* EMG_VIEW_H */
