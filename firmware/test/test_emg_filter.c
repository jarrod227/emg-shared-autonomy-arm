/* Host tests for the fixed-point sEMG band-pass.
 *
 * The same emg_filter.c that gets flashed is compiled here with host gcc,
 * because fixed-point arithmetic is where the mistakes live and stepping
 * through it over SWD is enormously slower than running `make check`.
 *
 * `--emit` writes golden.bin: an input sequence and the outputs this
 * implementation produced. tools/test_emg_filter_ref.py reruns the same input
 * through a float reference and checks the two agree.
 */

#include "emg_filter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Deterministic and simple enough that the Python side does not need to
 * reproduce it -- the emitted file carries the inputs as well as the
 * outputs. */
static int16_t golden_input(int index)
{
    const double angle = 6.283185307179586 * index;
    const double value = 900.0 * sin(angle * 80.0 / 2000.0)
                       + 400.0 * sin(angle * 250.0 / 2000.0)
                       + 600.0
                       + 60.0 * sin(angle * 3.0 / 2000.0);
    if (value > 2047.0) {
        return 2047;
    }
    if (value < -2048.0) {
        return -2048;
    }
    return (int16_t)value;
}

static void test_init_validation(void)
{
    emg_filter_t filter;
    const emg_biquad_coeffs_t section = {1 << EMG_FILTER_COEFF_BITS, 0, 0, 0, 0};

    CHECK(emg_filter_init(&filter, &section, 1) == true);
    CHECK(emg_filter_init(NULL, &section, 1) == false);
    CHECK(emg_filter_init(&filter, NULL, 1) == false);
    CHECK(emg_filter_init(&filter, &section, 0) == false);
    CHECK(emg_filter_init(&filter, &section, EMG_FILTER_MAX_SECTIONS + 1) == false);
}

static void test_unity_section_passes_samples_through(void)
{
    emg_filter_t filter;
    /* b0 = 1.0 in Q29, everything else zero: y[n] = x[n]. */
    const emg_biquad_coeffs_t section = {1 << EMG_FILTER_COEFF_BITS, 0, 0, 0, 0};
    const int16_t inputs[] = {0, 1, -1, 2047, -2048, 137};

    CHECK(emg_filter_init(&filter, &section, 1));
    for (size_t index = 0; index < sizeof(inputs) / sizeof(inputs[0]); index++) {
        CHECK(emg_filter_step(&filter, inputs[index]) == inputs[index]);
    }
}

static void test_band_pass_rejects_dc(void)
{
    emg_filter_t filter;

    CHECK(emg_filter_init(&filter, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    int32_t last = 0;
    for (int index = 0; index < 3000; index++) {
        last = emg_filter_step(&filter, 2000);
    }
    /* A constant input must settle to zero, and to exactly zero rather than a
     * residue: rounding bias would show up here first. */
    CHECK(last == 0);
}

static void test_band_pass_is_stable_after_an_impulse(void)
{
    emg_filter_t filter;
    int32_t peak = 0;

    CHECK(emg_filter_init(&filter, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    (void)emg_filter_step(&filter, 2047);
    for (int index = 0; index < 20000; index++) {
        const int32_t value = emg_filter_step(&filter, 0);
        const int32_t magnitude = value < 0 ? -value : value;
        if (magnitude > peak) {
            peak = magnitude;
        }
    }
    /* The tail must decay, not ring forever or grow. */
    CHECK(peak <= 2047);
    CHECK(emg_filter_step(&filter, 0) == 0);
}

static void test_reset_clears_history(void)
{
    emg_filter_t first;
    emg_filter_t second;

    CHECK(emg_filter_init(&first, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    CHECK(emg_filter_init(&second, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    for (int index = 0; index < 100; index++) {
        (void)emg_filter_step(&first, golden_input(index));
    }
    emg_filter_reset(&first);
    for (int index = 0; index < 50; index++) {
        CHECK(emg_filter_step(&first, golden_input(index)) ==
              emg_filter_step(&second, golden_input(index)));
    }
}

static void test_block_matches_step(void)
{
    emg_filter_t blockwise;
    emg_filter_t stepwise;
    int16_t input[64];
    int32_t output[64];

    for (int index = 0; index < 64; index++) {
        input[index] = golden_input(index);
    }
    CHECK(emg_filter_init(&blockwise, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    CHECK(emg_filter_init(&stepwise, emg_filter_bandpass_20_450_at_2000,
                          EMG_FILTER_BANDPASS_SECTIONS));
    emg_filter_block(&blockwise, input, output, 64);
    for (int index = 0; index < 64; index++) {
        CHECK(output[index] == emg_filter_step(&stepwise, input[index]));
    }
}

static int emit_golden(const char *path)
{
    emg_filter_t filter;
    FILE *file = fopen(path, "wb");
    int32_t count = GOLDEN_SAMPLES;

    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    if (!emg_filter_init(&filter, emg_filter_bandpass_20_450_at_2000,
                         EMG_FILTER_BANDPASS_SECTIONS)) {
        fclose(file);
        return 1;
    }
    fwrite(&count, sizeof(count), 1, file);
    for (int index = 0; index < count; index++) {
        const int16_t sample = golden_input(index);
        fwrite(&sample, sizeof(sample), 1, file);
    }
    emg_filter_reset(&filter);
    for (int index = 0; index < count; index++) {
        const int32_t value = emg_filter_step(&filter, golden_input(index));
        fwrite(&value, sizeof(value), 1, file);
    }
    fclose(file);
    printf("  wrote %s\n", path);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_golden(argv[2]);
    }

    printf("test_emg_filter\n");
    test_init_validation();
    test_unity_section_passes_samples_through();
    test_band_pass_rejects_dc();
    test_band_pass_is_stable_after_an_impulse();
    test_reset_clears_history();
    test_block_matches_step();

    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
