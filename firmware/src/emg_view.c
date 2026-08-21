#include "emg_view.h"

int8_t emg_view_direction(emg_command_t decision)
{
    switch (decision) {
    case EMG_COMMAND_NEXT_TARGET:
        return -1;
    default:
        return 0;
    }
}

uint16_t emg_view_activation(int32_t total_mav, int32_t threshold)
{
    if (threshold <= 0 || total_mav <= threshold) {
        return 0u;
    }
    /* span is computed from the threshold rather than stored, so a
     * reconfigured threshold moves the reference with it and the two can
     * never disagree. */
    const int64_t reference =
        (int64_t)threshold * EMG_VIEW_REFERENCE_NUM / EMG_VIEW_REFERENCE_DEN;
    const int64_t span = reference - (int64_t)threshold;
    if (span <= 0) {
        return 0u;
    }
    const int64_t above = (int64_t)total_mav - (int64_t)threshold;
    if (above >= span) {
        return 65535u;
    }
    /* Multiply before dividing: the other order truncates every value below
     * the reference to zero in integer arithmetic. */
    return (uint16_t)((above * 65535) / span);
}
