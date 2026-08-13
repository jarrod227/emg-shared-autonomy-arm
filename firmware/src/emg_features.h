/* Time-domain sEMG features over a sliding window, in integer arithmetic.
 *
 * MAV, RMS, waveform length, and zero crossings -- the four the Objective 3.5
 * classifier is specified to use. All four are amplitude or rate statistics,
 * which is why the ~6.8 us inter-channel skew of ADC1 scan mode does not
 * matter: none of them compares one channel's phase against another's.
 *
 * Like emg_filter.c this is pure logic with no HAL dependency, so the same
 * source is compiled into the firmware and into host tests.
 *
 * One window per channel. At 3 channels the state is about 2.5 KB of the
 * part's 20 KB.
 */

#ifndef EMG_FEATURES_H
#define EMG_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

/* 200 ms window, 50 ms hop, at the 2000 Hz sampling target. Changing the
 * sample rate means changing these together, or the window stops being
 * 200 ms and every trained model becomes invalid. */
#define EMG_FEATURES_WINDOW 400u
#define EMG_FEATURES_HOP 100u

typedef struct {
    int32_t mean_absolute_value; /* counts */
    int32_t root_mean_square;    /* counts */
    int32_t waveform_length;     /* counts, summed across the window */
    int32_t zero_crossings;      /* count across the window */
} emg_features_t;

typedef struct {
    /* Filtered samples are stored as int16. A 12-bit ADC centred by the
     * band-pass gives roughly +/-2048, so int16 leaves more than an order of
     * magnitude of headroom; `saturations` counts any sample that needed
     * clamping anyway, so the assumption fails loudly rather than by
     * quietly flattening peaks. */
    int16_t samples[EMG_FEATURES_WINDOW];
    uint16_t write_index;
    uint16_t filled;
    uint16_t since_hop;
    uint16_t zero_crossing_threshold;
    uint32_t saturations;
} emg_feature_window_t;

/* `zero_crossing_threshold` is the minimum sample-to-sample swing, in counts,
 * that a sign change must carry to be counted. Without it the resting noise
 * floor produces a large and meaningless crossing rate. It has no safe
 * default -- set it from the measured resting noise of this electrode
 * placement, not from a guess. */
bool emg_features_init(emg_feature_window_t *window,
                       uint16_t zero_crossing_threshold);

void emg_features_reset(emg_feature_window_t *window);

/* Add one filtered sample. Returns true, and fills `features`, only on a hop
 * boundary once the window has filled -- so the first result appears after
 * EMG_FEATURES_WINDOW samples, and one follows every EMG_FEATURES_HOP after
 * that. Partial windows never produce features: they would be computed over
 * fewer samples and silently differ in scale. */
bool emg_features_push(emg_feature_window_t *window, int32_t sample,
                       emg_features_t *features);

/* Compute over whatever is currently buffered, ignoring hop timing. Returns
 * false unless the window is full. */
bool emg_features_compute(const emg_feature_window_t *window,
                          emg_features_t *features);

/* floor(sqrt(value)). Exposed because the RMS definition depends on it and
 * the host reference has to match it exactly. */
uint32_t emg_isqrt(uint64_t value);

#endif /* EMG_FEATURES_H */
