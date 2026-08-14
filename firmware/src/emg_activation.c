/* See emg_activation.h. */

#include "emg_activation.h"

#include <stddef.h>

bool emg_activation_init(emg_activation_t *activation, uint16_t factor,
                         uint16_t baseline_shift)
{
    if (activation == NULL) {
        return false;
    }
    if (factor == 0u) {
        return false;
    }
    if (baseline_shift == 0u || baseline_shift > 8u) {
        return false;
    }
    activation->accumulator = 0;
    activation->factor = factor;
    activation->baseline_shift = baseline_shift;
    activation->has_baseline = false;
    return true;
}

int32_t emg_activation_baseline(const emg_activation_t *activation)
{
    if (activation == NULL || !activation->has_baseline) {
        return 0;
    }
    return activation->accumulator >> activation->baseline_shift;
}

emg_command_t emg_activation_apply(emg_activation_t *activation,
                                   emg_command_t prediction, bool valid,
                                   int32_t total_mav)
{
    if (activation == NULL) {
        return prediction;
    }
    if (!valid) {
        return prediction;
    }
    if (total_mav < 0) {
        total_mav = 0;
    } else if (total_mav > EMG_ACTIVATION_TOTAL_LIMIT) {
        total_mav = EMG_ACTIVATION_TOTAL_LIMIT;
    }

    if (prediction == EMG_COMMAND_REST) {
        if (!activation->has_baseline) {
            activation->accumulator =
                total_mav << activation->baseline_shift;
            activation->has_baseline = true;
        } else {
            /* acc += A - (acc >> s) keeps the sub-LSB residue inside the
             * accumulator, so a small persistent drift moves the baseline
             * instead of dying in a deadband. Every term stays
             * non-negative, so no negative right shifts occur. */
            activation->accumulator +=
                total_mav
                - (activation->accumulator >> activation->baseline_shift);
        }
        return EMG_COMMAND_REST;
    }

    if (!activation->has_baseline) {
        /* No rest observed yet: nothing to judge against. Fail open so a
         * cold-start ABORT stays reachable; ordinary events cannot fire
         * before the gate has seen rest anyway. */
        return prediction;
    }
    const int64_t threshold =
        (int64_t)emg_activation_baseline(activation)
        * (int64_t)activation->factor;
    if ((int64_t)total_mav < threshold) {
        /* The shape of a gesture without the activation of one. */
        return EMG_COMMAND_REST;
    }
    return prediction;
}
