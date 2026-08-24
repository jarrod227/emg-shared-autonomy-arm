#include "emg_view.h"

int8_t emg_view_direction(emg_command_t decision)
{
    switch (decision) {
    case EMG_COMMAND_NEXT_TARGET:
        return -1;
    case EMG_COMMAND_ULNAR:
        return 1;
    default:
        return 0;
    }
}

int32_t emg_view_reference(int8_t direction, int32_t threshold,
                           int32_t reference_left, int32_t reference_right)
{
    if (direction == 0) {
        return 0;
    }
    const int32_t measured = (direction < 0) ? reference_left : reference_right;
    if (measured > 0) {
        return measured;
    }
    /* No calibrated ceiling for this direction. Deriving one from the
     * threshold is what this replaced, and it is wrong in a known direction
     * -- too low, so the wearer saturates -- but it keeps the axis steerable
     * on an uncalibrated board instead of freezing it. */
    if (threshold <= 0) {
        return 0;
    }
    return (int32_t)((int64_t)threshold * EMG_VIEW_REFERENCE_NUM
                     / EMG_VIEW_REFERENCE_DEN);
}

uint16_t emg_view_activation(int32_t total_mav, int32_t threshold,
                             int32_t reference)
{
    if (threshold <= 0 || total_mav <= threshold) {
        return 0u;
    }
    const int64_t span = (int64_t)reference - (int64_t)threshold;
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
