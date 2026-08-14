/* See emg_gate.h.
 *
 * The order of the checks in emg_gate_push is itself the specification: it
 * mirrors EventGate.push in firmware/tools/emg_event_gate_replay.py line for
 * line, because that is the implementation the frozen counts were validated
 * with. Reordering the branches changes behaviour even when every count is
 * unchanged -- for example, clearing the rest run before the hold-off check is
 * what makes a twitch reset re-arming. test_emg_gate.c emits a fixture that
 * the Python gate re-derives, so a divergence fails a test rather than
 * becoming a field report.
 */

#include "emg_gate.h"

#include <stddef.h>

void emg_gate_default_config(emg_gate_config_t *config)
{
    if (config == NULL) {
        return;
    }
    config->stable_windows = EMG_GATE_STABLE_WINDOWS;
    config->rest_rearm_windows = EMG_GATE_REST_REARM_WINDOWS;
    config->refractory_windows = EMG_GATE_REFRACTORY_WINDOWS;
    config->abort_stable_windows = EMG_GATE_ABORT_STABLE_WINDOWS;
    config->onset_holdoff_windows = EMG_GATE_ONSET_HOLDOFF_WINDOWS;
}

bool emg_gate_init(emg_gate_t *gate, const emg_gate_config_t *config)
{
    if (gate == NULL || config == NULL) {
        return false;
    }
    if (config->stable_windows == 0u
        || config->rest_rearm_windows == 0u
        || config->abort_stable_windows == 0u) {
        return false;
    }
    gate->config = *config;
    emg_gate_invalidate(gate);
    return true;
}

void emg_gate_invalidate(emg_gate_t *gate)
{
    if (gate == NULL) {
        return;
    }
    gate->candidate_run = 0u;
    gate->rest_run = 0u;
    gate->abort_run = 0u;
    gate->refractory = 0u;
    gate->onset_holdoff = 0u;
    gate->candidate = EMG_COMMAND_REST;
    gate->has_candidate = false;
    gate->armed = false;
    gate->abort_latched = false;
    gate->resting = true;
}

bool emg_gate_push(emg_gate_t *gate, emg_command_t prediction, bool valid,
                   emg_command_t *event)
{
    if (gate == NULL || event == NULL) {
        return false;
    }
    if (prediction != EMG_COMMAND_REST
        && prediction != EMG_COMMAND_NEXT_TARGET
        && prediction != EMG_COMMAND_CONFIRM
        && prediction != EMG_COMMAND_ABORT) {
        emg_gate_invalidate(gate);
        return false;
    }
    if (!valid) {
        emg_gate_invalidate(gate);
        return false;
    }

    /* The refractory period is elapsed time, so it keeps counting down through
     * a hold-off and through rest alike. */
    if (gate->refractory > 0u) {
        gate->refractory--;
    }

    if (prediction == EMG_COMMAND_REST) {
        gate->resting = true;
        gate->onset_holdoff = 0u;
        gate->has_candidate = false;
        gate->candidate_run = 0u;
        gate->abort_run = 0u;
        if (gate->rest_run < UINT16_MAX) {
            gate->rest_run++;
        }
        if (gate->rest_run >= gate->config.rest_rearm_windows
            && gate->refractory == 0u) {
            gate->armed = true;
            gate->abort_latched = false;
        }
        return false;
    }

    /* Any contraction ends the rest run, including one that is about to be
     * discarded by the hold-off: a twitch means the muscle was not resting. */
    gate->rest_run = 0u;
    if (gate->resting) {
        gate->resting = false;
        gate->onset_holdoff = gate->config.onset_holdoff_windows;
    }
    if (gate->onset_holdoff > 0u) {
        gate->onset_holdoff--;
        gate->has_candidate = false;
        gate->candidate_run = 0u;
        gate->abort_run = 0u;
        return false;
    }

    if (prediction == EMG_COMMAND_ABORT) {
        /* ABORT bypasses arming and the refractory period -- a stop must not
         * wait on a gesture that just fired -- but it latches until rest, so a
         * held abort still emits once. */
        gate->has_candidate = false;
        gate->candidate_run = 0u;
        if (gate->abort_run < UINT16_MAX) {
            gate->abort_run++;
        }
        if (gate->abort_run >= gate->config.abort_stable_windows
            && !gate->abort_latched) {
            gate->abort_latched = true;
            gate->armed = false;
            gate->refractory = gate->config.refractory_windows;
            *event = EMG_COMMAND_ABORT;
            return true;
        }
        return false;
    }

    gate->abort_run = 0u;
    if (!gate->armed || gate->refractory > 0u) {
        gate->has_candidate = false;
        gate->candidate_run = 0u;
        return false;
    }
    if (gate->has_candidate && prediction == gate->candidate) {
        if (gate->candidate_run < UINT16_MAX) {
            gate->candidate_run++;
        }
    } else {
        gate->candidate = prediction;
        gate->has_candidate = true;
        gate->candidate_run = 1u;
    }
    if (gate->candidate_run < gate->config.stable_windows) {
        return false;
    }

    *event = gate->candidate;
    gate->armed = false;
    gate->has_candidate = false;
    gate->candidate_run = 0u;
    gate->refractory = gate->config.refractory_windows;
    return true;
}
