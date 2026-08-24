/* Host checks for the proportional view output, plus a fixture the Python
 * reference is compared against. The C and the reference must agree exactly,
 * not approximately: the integer truncation is part of the answer, and a
 * host tool that predicts a different activation than the board emits is
 * worse than no prediction at all. */

#include "emg_view.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #condition);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static void test_only_the_assigned_gesture_steers(void)
{
    CHECK(emg_view_direction(EMG_COMMAND_NEXT_TARGET) == -1);
    CHECK(emg_view_direction(EMG_COMMAND_REST) == 0);
    CHECK(emg_view_direction(EMG_COMMAND_CONFIRM) == 0);
    CHECK(emg_view_direction(EMG_COMMAND_ABORT) == 0);
    CHECK(emg_view_direction(EMG_COMMAND_ULNAR) == 1);
}

static int32_t fallback(int32_t threshold)
{
    return threshold * EMG_VIEW_REFERENCE_NUM / EMG_VIEW_REFERENCE_DEN;
}

static void test_the_reference_is_per_direction(void)
{
    /* Measured 2026-08-23 in one capture on one donning: wrist extension
     * 4.19x the session threshold, ulnar deviation 5.51x. No constant spans
     * that, and the constant that used to try is 3, below both. */
    CHECK(emg_view_reference(-1, 55, 231, 303) == 231);
    CHECK(emg_view_reference(1, 55, 231, 303) == 303);
    /* Missing on one side falls back on that side only. */
    CHECK(emg_view_reference(-1, 55, 0, 303) == fallback(55));
    CHECK(emg_view_reference(1, 55, 231, 0) == fallback(55));
    /* No direction means no span to take a fraction of. */
    CHECK(emg_view_reference(0, 55, 231, 303) == 0);
    CHECK(emg_view_activation(400, 55, emg_view_reference(0, 55, 231, 303))
          == 0);
}

static void test_a_reference_inside_the_threshold_gives_no_deflection(void)
{
    CHECK(emg_view_activation(400, 100, 100) == 0);
    CHECK(emg_view_activation(400, 100, 60) == 0);
}

static void test_a_measured_reference_unsaturates_the_same_effort(void)
{
    /* threshold 55 with a NEXT_TARGET reference of 231, where the fallback
     * would be 165 and would clip. */
    CHECK(emg_view_activation(200, 55, fallback(55)) == 65535);
    const uint16_t measured = emg_view_activation(200, 55, 231);
    CHECK(measured < 65535);
    CHECK(measured > 45000);
}

static void test_zero_at_and_below_the_threshold(void)
{
    CHECK(emg_view_activation(0, 100, fallback(100)) == 0);
    CHECK(emg_view_activation(99, 100, fallback(100)) == 0);
    CHECK(emg_view_activation(100, 100, fallback(100)) == 0);
    CHECK(emg_view_activation(101, 100, fallback(100)) > 0);
    /* A threshold of zero would divide by a zero span. */
    CHECK(emg_view_activation(500, 0, fallback(0)) == 0);
    CHECK(emg_view_activation(500, -10, fallback(-10)) == 0);
}

static void test_saturates_rather_than_wrapping(void)
{
    const int32_t threshold = 100;
    const int32_t reference =
        threshold * EMG_VIEW_REFERENCE_NUM / EMG_VIEW_REFERENCE_DEN;
    CHECK(emg_view_activation(reference, threshold, reference) == 65535);
    CHECK(emg_view_activation(reference * 10, threshold, reference) == 65535);
    /* The largest total the features can produce must not wrap either. */
    CHECK(emg_view_activation(3 * 32767, threshold, reference) == 65535);
}

static void test_monotonic_and_multiplies_before_dividing(void)
{
    const int32_t threshold = 1000;
    uint16_t previous = 0;
    for (int32_t total = 1000; total <= 3000; total++) {
        const uint16_t value =
            emg_view_activation(total, threshold, fallback(threshold));
        CHECK(value >= previous);
        previous = value;
    }
    /* Dividing first truncates everything below the reference to zero, which
     * reads as a dead channel rather than as an arithmetic mistake. */
    const int32_t reference =
        threshold * EMG_VIEW_REFERENCE_NUM / EMG_VIEW_REFERENCE_DEN;
    CHECK(emg_view_activation(threshold + (reference - threshold) / 100,
                              threshold, reference) > 500);
}

static void emit_fixture(const char *path)
{
    static const int32_t thresholds[] = {64, 68, 96, 110, 164, 1000};
    const int32_t steps = 240;
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        printf("  FAIL cannot open %s\n", path);
        failures++;
        return;
    }
    const int32_t count = (int32_t)(sizeof(thresholds) / sizeof(thresholds[0]));
    fwrite(&count, sizeof(count), 1, file);
    fwrite(&steps, sizeof(steps), 1, file);
    fwrite(thresholds, sizeof(thresholds[0]), (size_t)count, file);
    for (int32_t i = 0; i < count; i++) {
        for (int32_t step = 0; step < steps; step++) {
            const int32_t total = step * 5;
            const uint16_t value = emg_view_activation(
                total, thresholds[i], fallback(thresholds[i]));
            fwrite(&value, sizeof(value), 1, file);
        }
    }
    fclose(file);
    printf("  wrote %s (%d thresholds x %d totals)\n", path, count, steps);
}

int main(int argc, char **argv)
{
    printf("test_emg_view\n");
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        emit_fixture(argv[2]);
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    test_only_the_assigned_gesture_steers();
    test_the_reference_is_per_direction();
    test_a_reference_inside_the_threshold_gives_no_deflection();
    test_a_measured_reference_unsaturates_the_same_effort();
    test_zero_at_and_below_the_threshold();
    test_saturates_rather_than_wrapping();
    test_monotonic_and_multiplies_before_dividing();
    if (failures == 0) {
        printf("  all checks passed\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
