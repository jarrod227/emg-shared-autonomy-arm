/* See emg_features.h. */

#include "emg_features.h"

#include <string.h>

#define INT16_LOWEST (-32768)
#define INT16_HIGHEST 32767

uint32_t emg_isqrt(uint64_t value)
{
    /* Bit-by-bit integer square root: no division and no floating point, and
     * a fixed iteration count, so it costs the same every window. Returns
     * floor(sqrt(value)), which is what Python's math.isqrt returns, so the
     * host reference can match this bit for bit rather than approximately. */
    uint64_t remainder = value;
    uint64_t result = 0;
    uint64_t bit = (uint64_t)1 << 62;

    while (bit > remainder) {
        bit >>= 2;
    }
    while (bit != 0u) {
        if (remainder >= result + bit) {
            remainder -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)result;
}

bool emg_features_init(emg_feature_window_t *window,
                       uint16_t zero_crossing_threshold)
{
    if (window == NULL) {
        return false;
    }
    memset(window, 0, sizeof(*window));
    window->zero_crossing_threshold = zero_crossing_threshold;
    return true;
}

void emg_features_reset(emg_feature_window_t *window)
{
    if (window == NULL) {
        return;
    }
    memset(window->samples, 0, sizeof(window->samples));
    window->write_index = 0u;
    window->filled = 0u;
    window->since_hop = 0u;
    window->saturations = 0u;
}

static int16_t clamp_to_int16(emg_feature_window_t *window, int32_t sample)
{
    if (sample > INT16_HIGHEST) {
        window->saturations++;
        return (int16_t)INT16_HIGHEST;
    }
    if (sample < INT16_LOWEST) {
        window->saturations++;
        return (int16_t)INT16_LOWEST;
    }
    return (int16_t)sample;
}

/* Oldest-first index into the ring. Chronological order matters: waveform
 * length and zero crossings are both defined on consecutive differences, so
 * reading the buffer in storage order would splice the newest sample against
 * the oldest once per window. */
static int32_t sample_at(const emg_feature_window_t *window, uint16_t offset)
{
    const uint16_t index =
        (uint16_t)((window->write_index + offset) % EMG_FEATURES_WINDOW);
    return (int32_t)window->samples[index];
}

bool emg_features_compute(const emg_feature_window_t *window,
                          emg_features_t *features)
{
    if (window == NULL || features == NULL) {
        return false;
    }
    if (window->filled < EMG_FEATURES_WINDOW) {
        return false;
    }

    uint64_t absolute_sum = 0;
    uint64_t square_sum = 0;
    uint64_t length_sum = 0;
    uint32_t crossings = 0;
    int32_t previous = sample_at(window, 0);

    absolute_sum += (uint64_t)(previous < 0 ? -previous : previous);
    square_sum += (uint64_t)((int64_t)previous * previous);

    for (uint16_t offset = 1; offset < EMG_FEATURES_WINDOW; offset++) {
        const int32_t current = sample_at(window, offset);
        const int32_t difference = current - previous;
        const int32_t magnitude = difference < 0 ? -difference : difference;

        absolute_sum += (uint64_t)(current < 0 ? -current : current);
        square_sum += (uint64_t)((int64_t)current * current);
        length_sum += (uint64_t)magnitude;

        /* A sign change only counts if the swing carries enough amplitude.
         * Without that gate the resting noise floor dithers around zero and
         * produces a large crossing rate that says nothing about the muscle.
         * Exact zeros are treated as neither sign, so they never form a
         * crossing on their own. */
        if (((current > 0 && previous < 0) || (current < 0 && previous > 0))
            && magnitude >= (int32_t)window->zero_crossing_threshold) {
            crossings++;
        }
        previous = current;
    }

    features->mean_absolute_value =
        (int32_t)(absolute_sum / EMG_FEATURES_WINDOW);
    features->root_mean_square =
        (int32_t)emg_isqrt(square_sum / EMG_FEATURES_WINDOW);
    features->waveform_length = (int32_t)length_sum;
    features->zero_crossings = (int32_t)crossings;
    return true;
}

bool emg_features_push(emg_feature_window_t *window, int32_t sample,
                       emg_features_t *features)
{
    if (window == NULL || features == NULL) {
        return false;
    }

    window->samples[window->write_index] = clamp_to_int16(window, sample);
    window->write_index =
        (uint16_t)((window->write_index + 1u) % EMG_FEATURES_WINDOW);
    if (window->filled < EMG_FEATURES_WINDOW) {
        window->filled++;
    }
    if (window->since_hop < EMG_FEATURES_HOP) {
        window->since_hop++;
    }

    if (window->filled < EMG_FEATURES_WINDOW
        || window->since_hop < EMG_FEATURES_HOP) {
        return false;
    }
    window->since_hop = 0u;
    return emg_features_compute(window, features);
}
