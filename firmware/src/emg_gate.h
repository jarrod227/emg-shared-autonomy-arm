/* Fail-closed discrete-event gate over per-hop classifier decisions.
 *
 * The classifier answers "what does this 200 ms window look like" every 50 ms.
 * That is not an intent: one gesture produces about forty decisions, and a
 * held gesture produces them indefinitely. This module turns that stream into
 * at most one event per gesture, and refuses to emit when the evidence is
 * unreadable rather than guessing.
 *
 * Four independent rules, each buying a different guarantee:
 *
 *   stable run     N agreeing decisions before an event; rejects jitter.
 *   REST re-arm    a gesture cannot fire again until rest is observed, so
 *                  holding one does not repeat it.
 *   refractory     a settling period after an event.
 *   onset hold-off decisions are discarded for a fixed count after leaving
 *                  rest, because a 200 ms window straddling rest and
 *                  contraction is a mixture the classifier never trained on.
 *
 * The hold-off is not a longer stable run and cannot be replaced by one.
 * Measured on the validation session, one ABORT trial had a sixteen-decision
 * block of CONFIRM at its onset and a longest correct run of six: any
 * threshold that rejects the wrong block also rejects the right one. Length
 * cannot separate them; position can.
 *
 * Pure logic with no HAL dependency, so the same source is compiled into the
 * firmware and into host tests, and checked against the Python EventGate that
 * the parameters were actually validated with.
 */

#ifndef EMG_GATE_H
#define EMG_GATE_H

#include "emg_packet.h"

#include <stdbool.h>
#include <stdint.h>

/* Frozen 2026-08-14 by an independent event-gate session: fitted on the first
 * donning, with these counts fixed beforehand on a different session, a third
 * donning produced 9/9 clean events with no missed, wrong, duplicate,
 * REST-false, or off-trial event. They are a measurement result, not tuning
 * knobs -- changing one invalidates that session and needs a new one. Kept in
 * step with VALIDATED_GATE in firmware/tools/emg_event_gate_replay.py; see
 * docs/objective35_classifier_log.md. Counts are in 50 ms feature hops. */
#define EMG_GATE_STABLE_WINDOWS        5u
#define EMG_GATE_REST_REARM_WINDOWS    4u
#define EMG_GATE_REFRACTORY_WINDOWS    5u
#define EMG_GATE_ABORT_STABLE_WINDOWS  1u
#define EMG_GATE_ONSET_HOLDOFF_WINDOWS 12u

typedef struct {
    uint16_t stable_windows;
    uint16_t rest_rearm_windows;
    uint16_t refractory_windows;
    uint16_t abort_stable_windows;
    uint16_t onset_holdoff_windows;
} emg_gate_config_t;

typedef struct {
    emg_gate_config_t config;
    uint16_t candidate_run;
    uint16_t rest_run;
    uint16_t abort_run;
    uint16_t refractory;
    uint16_t onset_holdoff;
    emg_command_t candidate;
    bool has_candidate;
    bool armed;
    bool abort_latched;
    bool resting;
} emg_gate_t;

/* The frozen configuration above. */
void emg_gate_default_config(emg_gate_config_t *config);

/* Rejects a zero stable/re-arm/abort count, which would emit on no evidence.
 * A zero refractory or hold-off is legal and means that rule is off. */
bool emg_gate_init(emg_gate_t *gate, const emg_gate_config_t *config);

/* Drop all accumulated evidence. Valid REST is required before an ordinary
 * event can fire again. An interrupted stream is not evidence that the muscle
 * is resting, so the next contraction still counts as an onset. */
void emg_gate_invalidate(emg_gate_t *gate);

/* Feed one classifier decision. `valid` must be false whenever any sample in
 * the window was missing or its electrode was detached. Returns true when an
 * event fires, writing it to `event`; `event` is untouched otherwise. REST is
 * never an event -- absence of intent is not an intent. */
bool emg_gate_push(emg_gate_t *gate, emg_command_t prediction, bool valid,
                   emg_command_t *event);

#endif /* EMG_GATE_H */
