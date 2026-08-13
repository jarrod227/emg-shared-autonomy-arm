/* Host tests for the sEMG feature extractor.
 *
 * Everything here is integer arithmetic, so the host reference can be checked
 * for exact equality rather than within a tolerance -- unlike the filter,
 * where fixed point is compared against float.
 *
 * `--emit` writes features.bin for tools/test_emg_features_ref.py.
 */

#include "emg_features.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #condition);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

#define GOLDEN_SAMPLES 2000
#define ZC_THRESHOLD 10

static int16_t golden_input(int index)
{
    const double angle = 6.283185307179586 * index;
    /* An 80 Hz carrier whose amplitude swells and fades, which is roughly how
     * a contraction looks, plus a little noise so zero crossings are not
     * trivially periodic. */
    const double envelope = 200.0 + 700.0 * fabs(sin(angle * 0.7 / 2000.0));
    const double value = envelope * sin(angle * 80.0 / 2000.0)
                       + 40.0 * sin(angle * 317.0 / 2000.0);
    return (int16_t)value;
}

static void test_isqrt(void)
{
    CHECK(emg_isqrt(0) == 0u);
    CHECK(emg_isqrt(1) == 1u);
    CHECK(emg_isqrt(3) == 1u);
    CHECK(emg_isqrt(4) == 2u);
    CHECK(emg_isqrt(99) == 9u);
    CHECK(emg_isqrt(100) == 10u);
    CHECK(emg_isqrt(101) == 10u);
    CHECK(emg_isqrt((uint64_t)1 << 32) == 65536u);
    CHECK(emg_isqrt(((uint64_t)1 << 32) - 1u) == 65535u);
    /* Every value the RMS path can reach: the largest square sum is
     * WINDOW * 32767^2, divided by WINDOW, so 32767 is the ceiling. */
    CHECK(emg_isqrt(32767u * 32767u) == 32767u);
}

static void test_init_validation(void)
{
    emg_feature_window_t window;

    CHECK(emg_features_init(&window, ZC_THRESHOLD) == true);
    CHECK(emg_features_init(NULL, ZC_THRESHOLD) == false);
    CHECK(window.zero_crossing_threshold == ZC_THRESHOLD);
    CHECK(window.saturations == 0u);
}

static void test_no_features_until_the_window_fills(void)
{
    emg_feature_window_t window;
    emg_features_t features;
    int emitted = 0;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    /* A partial window would be averaged over fewer samples and silently
     * differ in scale from every later one. */
    CHECK(emg_features_compute(&window, &features) == false);
    for (unsigned index = 0; index < EMG_FEATURES_WINDOW - 1u; index++) {
        if (emg_features_push(&window, 100, &features)) {
            emitted++;
        }
    }
    CHECK(emitted == 0);
    CHECK(emg_features_push(&window, 100, &features) == true);
}

static void test_hop_spacing(void)
{
    emg_feature_window_t window;
    emg_features_t features;
    int first_emit = -1;
    int second_emit = -1;
    int count = 0;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    for (int index = 0; index < 1000; index++) {
        if (emg_features_push(&window, golden_input(index), &features)) {
            if (first_emit < 0) {
                first_emit = index;
            } else if (second_emit < 0) {
                second_emit = index;
            }
            count++;
        }
    }
    CHECK(first_emit == (int)EMG_FEATURES_WINDOW - 1);
    CHECK(second_emit - first_emit == (int)EMG_FEATURES_HOP);
    CHECK(count == 1 + (1000 - (int)EMG_FEATURES_WINDOW) / (int)EMG_FEATURES_HOP);
}

static void test_constant_signal(void)
{
    emg_feature_window_t window;
    emg_features_t features;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    for (unsigned index = 0; index < EMG_FEATURES_WINDOW; index++) {
        (void)emg_features_push(&window, -250, &features);
    }
    CHECK(emg_features_compute(&window, &features));
    CHECK(features.mean_absolute_value == 250);
    CHECK(features.root_mean_square == 250);
    CHECK(features.waveform_length == 0);
    CHECK(features.zero_crossings == 0);
}

static void test_square_wave_has_exact_features(void)
{
    emg_feature_window_t window;
    emg_features_t features;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    for (unsigned index = 0; index < EMG_FEATURES_WINDOW; index++) {
        (void)emg_features_push(&window, (index % 2u) ? -100 : 100, &features);
    }
    CHECK(emg_features_compute(&window, &features));
    CHECK(features.mean_absolute_value == 100);
    CHECK(features.root_mean_square == 100);
    /* 399 transitions of 200 counts each. */
    CHECK(features.waveform_length == 399 * 200);
    CHECK(features.zero_crossings == 399);
}

static void test_threshold_suppresses_noise_crossings(void)
{
    emg_feature_window_t quiet;
    emg_feature_window_t ungated;
    emg_features_t features;

    CHECK(emg_features_init(&quiet, 10));
    CHECK(emg_features_init(&ungated, 0));
    for (unsigned index = 0; index < EMG_FEATURES_WINDOW; index++) {
        const int32_t dither = (index % 2u) ? -2 : 2;
        (void)emg_features_push(&quiet, dither, &features);
        (void)emg_features_push(&ungated, dither, &features);
    }
    /* A 4-count swing must not register against a 10-count threshold, but
     * does when the gate is disabled -- which is the failure mode the
     * threshold exists to prevent. */
    CHECK(emg_features_compute(&quiet, &features));
    CHECK(features.zero_crossings == 0);
    CHECK(emg_features_compute(&ungated, &features));
    CHECK(features.zero_crossings == 399);
}

static void test_saturation_is_counted(void)
{
    emg_feature_window_t window;
    emg_features_t features;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    (void)emg_features_push(&window, 40000, &features);
    (void)emg_features_push(&window, -40000, &features);
    (void)emg_features_push(&window, 1000, &features);
    CHECK(window.saturations == 2u);
    CHECK(window.samples[0] == 32767);
    CHECK(window.samples[1] == -32768);
}

static void test_reset_clears_history(void)
{
    emg_feature_window_t window;
    emg_features_t features;

    CHECK(emg_features_init(&window, ZC_THRESHOLD));
    for (unsigned index = 0; index < EMG_FEATURES_WINDOW; index++) {
        (void)emg_features_push(&window, 500, &features);
    }
    emg_features_reset(&window);
    CHECK(emg_features_compute(&window, &features) == false);
    CHECK(window.filled == 0u);
    CHECK(window.saturations == 0u);
}

static int emit_golden(const char *path)
{
    emg_feature_window_t window;
    emg_features_t features;
    FILE *file = fopen(path, "wb");
    int32_t count = GOLDEN_SAMPLES;
    int32_t threshold = ZC_THRESHOLD;
    int32_t emitted = 0;
    long emitted_position;

    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    (void)emg_features_init(&window, ZC_THRESHOLD);
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&threshold, sizeof(threshold), 1, file);
    emitted_position = ftell(file);
    fwrite(&emitted, sizeof(emitted), 1, file);
    for (int index = 0; index < count; index++) {
        const int16_t sample = golden_input(index);
        fwrite(&sample, sizeof(sample), 1, file);
    }
    for (int index = 0; index < count; index++) {
        if (emg_features_push(&window, golden_input(index), &features)) {
            fwrite(&features.mean_absolute_value, 4, 1, file);
            fwrite(&features.root_mean_square, 4, 1, file);
            fwrite(&features.waveform_length, 4, 1, file);
            fwrite(&features.zero_crossings, 4, 1, file);
            emitted++;
        }
    }
    fseek(file, emitted_position, SEEK_SET);
    fwrite(&emitted, sizeof(emitted), 1, file);
    fclose(file);
    printf("  wrote %s (%d feature sets)\n", path, emitted);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_golden(argv[2]);
    }

    printf("test_emg_features\n");
    test_isqrt();
    test_init_validation();
    test_no_features_until_the_window_fills();
    test_hop_spacing();
    test_constant_signal();
    test_square_wave_has_exact_features();
    test_threshold_suppresses_noise_crossings();
    test_saturation_is_counted();
    test_reset_clears_history();

    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
