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
 * The reference level is the open problem. Activation is a fraction of the
 * span between the threshold and whatever the wearer can actually produce for
 * the driving gesture, and that ceiling has to be measured per donning and per
 * direction -- measured 2026-08-20, one donning's gesture maxima ran 118 to
 * 193 against a threshold of 68, and reference/threshold across three donnings
 * ran 2.38, 3.34 and 2.47. A 40% spread is not something a compile-time
 * constant can stand in for, so the constant below is interim and exists to
 * make a controllability experiment possible before the protocol grows a field
 * to carry a measured one. It sets the gain, not whether the signal is steady
 * enough to steer with, and it is the latter the experiment is for.
 */

#ifndef EMG_VIEW_H
#define EMG_VIEW_H

#include "emg_packet.h"

#include <stdint.h>

/* Interim, see above. reference = threshold * NUM / DEN. */
#define EMG_VIEW_REFERENCE_NUM 5
#define EMG_VIEW_REFERENCE_DEN 2

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
uint16_t emg_view_activation(int32_t total_mav, int32_t threshold);

#endif /* EMG_VIEW_H */
