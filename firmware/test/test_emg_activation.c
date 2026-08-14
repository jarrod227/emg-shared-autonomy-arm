/* Host tests for the same activation threshold linked into the firmware.
 *
 * The hand-written cases pin each rule, including the deliberate policy
 * choices — ABORT is judged like every other class, and rest windows update
 * the baseline even when loud — so a change to either shows up as a failing
 * test rather than as a silent behaviour shift. The emitted fixture runs a
 * long sequence built from the measured shapes and is replayed through the
 * bit-exact Python mirror by test_emg_activation_ref.py, which must agree
 * decision for decision and baseline for baseline.
 */

#include "emg_activation.h"

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

/* The hand-written cases below pin factor 5 / shift 4 deliberately, and no
 * longer track the candidate constants: the arithmetic in their comments
 * (29 -> threshold 145) is written against these, and a case that silently
 * re-derived its own expected values whenever the candidate moved would
 * stop testing the mechanism. test_candidate_constants_are_what_was_swept
 * covers the candidate itself. */
static void init_fixed(emg_activation_t *activation)
{
    CHECK(emg_activation_init(activation, 5u, 4u));
}

static emg_command_t judge(emg_activation_t *activation,
                           emg_command_t prediction, int32_t total_mav)
{
    return emg_activation_apply(activation, prediction, true, total_mav);
}

static void test_init_rejects_configs_that_disable_protection(void)
{
    emg_activation_t activation;

    CHECK(!emg_activation_init(NULL, 5u, 4u));
    CHECK(!emg_activation_init(&activation, 0u, 4u));
    CHECK(!emg_activation_init(&activation, 5u, 0u));
    CHECK(!emg_activation_init(&activation, 5u, 9u));
    CHECK(emg_activation_init(&activation, 5u, 8u));
    CHECK(emg_activation_init(&activation, 1u, 1u));
}

static void test_everything_passes_until_rest_seeds_the_baseline(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    /* A gesture already under way at power-on: nothing to judge against,
     * and a cold-start stop must stay reachable. */
    CHECK(judge(&activation, EMG_COMMAND_ABORT, 40) == EMG_COMMAND_ABORT);
    CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 55)
          == EMG_COMMAND_NEXT_TARGET);
    CHECK(emg_activation_baseline(&activation) == 0);
}

static void test_low_activation_shapes_become_rest_after_seeding(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    CHECK(emg_activation_baseline(&activation) == 29);

    /* The measured preparatory band, and the exact threshold boundary:
     * 5 x 29 = 145, suppression is strictly-below. */
    CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 70)
          == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_CONFIRM, 144) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_CONFIRM, 145)
          == EMG_COMMAND_CONFIRM);
    CHECK(judge(&activation, EMG_COMMAND_CONFIRM, 600)
          == EMG_COMMAND_CONFIRM);
}

static void test_threshold_scales_with_the_measured_baseline(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    /* A louder donning: donning B rested at about twice donning A. An
     * absolute floor tuned on 29 would pass 200 here; the relative one
     * does not. */
    CHECK(judge(&activation, EMG_COMMAND_REST, 60) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 200)
          == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 300)
          == EMG_COMMAND_NEXT_TARGET);
}

static void test_suppressed_windows_never_feed_the_baseline(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    /* A long sub-threshold movement: if its 70-count windows fed the EMA,
     * the baseline would crawl toward 70 and the threshold toward 350. */
    for (int index = 0; index < 50; index++) {
        CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 70)
              == EMG_COMMAND_REST);
    }
    CHECK(emg_activation_baseline(&activation) == 29);
    /* 200 passes against the true baseline and would have been suppressed
     * against the crawled one. */
    CHECK(judge(&activation, EMG_COMMAND_CONFIRM, 200)
          == EMG_COMMAND_CONFIRM);
}

static void test_baseline_tracks_drift_without_a_deadband(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    /* A plain floored EMA never moves for steps smaller than 2^shift; the
     * accumulator formulation must converge exactly. */
    for (int index = 0; index < 400; index++) {
        (void)judge(&activation, EMG_COMMAND_REST, 35);
    }
    CHECK(emg_activation_baseline(&activation) == 35);
    for (int index = 0; index < 400; index++) {
        (void)judge(&activation, EMG_COMMAND_REST, 20);
    }
    CHECK(emg_activation_baseline(&activation) == 20);
}

static void test_rest_windows_update_the_baseline_even_when_loud(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    /* The known limitation, pinned: a transition window that classifies as
     * REST with elevated amplitude does move the baseline. 464 + 300 - 29
     * = 735, and 735 >> 4 = 45. If the seed rule ever changes, this is the
     * test that says so. */
    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_REST, 300) == EMG_COMMAND_REST);
    CHECK(emg_activation_baseline(&activation) == 45);
}

static void test_invalid_windows_change_nothing(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    CHECK(emg_activation_apply(&activation, EMG_COMMAND_NEXT_TARGET, false,
                               600) == EMG_COMMAND_NEXT_TARGET);
    CHECK(emg_activation_apply(&activation, EMG_COMMAND_REST, false, 500)
          == EMG_COMMAND_REST);
    CHECK(emg_activation_baseline(&activation) == 29);
}

static void test_abort_is_judged_like_every_other_class(void)
{
    emg_activation_t activation;
    init_fixed(&activation);

    /* Deliberate policy: the measured ABORT trials ran 15-22x rest, so a
     * threshold leaves real margin at any swept factor, and a one-class
     * exemption would reopen the preparatory-movement defect for ABORT. */
    CHECK(judge(&activation, EMG_COMMAND_REST, 29) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_ABORT, 70) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_ABORT, 400) == EMG_COMMAND_ABORT);
}

static void test_corrupt_totals_cannot_poison_the_accumulator(void)
{
    emg_activation_t activation;

    init_fixed(&activation);
    CHECK(judge(&activation, EMG_COMMAND_REST, 2147483647) ==
          EMG_COMMAND_REST);
    CHECK(emg_activation_baseline(&activation)
          == EMG_ACTIVATION_TOTAL_LIMIT);

    init_fixed(&activation);
    CHECK(judge(&activation, EMG_COMMAND_REST, -100) == EMG_COMMAND_REST);
    CHECK(emg_activation_baseline(&activation) == 0);
}

static void test_candidate_constants_are_what_was_swept(void)
{
    /* The joint sweep over the defect recording and the frozen event-gate
     * session passed K = 2..5 and broke at 6; 3 was taken because it clears
     * the preparatory movement outright instead of by fragmenting it. These
     * are candidates awaiting an independent session, so the assertion is
     * here to make a change deliberate, not to claim they are validated. */
    CHECK(EMG_ACTIVATION_FACTOR == 3u);
    CHECK(EMG_ACTIVATION_BASELINE_SHIFT == 4u);

    /* The measured numbers the choice rests on: a 35-count rest baseline,
     * a preparatory movement peaking at 79, and a fist reaching 736. */
    emg_activation_t activation;
    CHECK(emg_activation_init(&activation, EMG_ACTIVATION_FACTOR,
                              EMG_ACTIVATION_BASELINE_SHIFT));
    CHECK(judge(&activation, EMG_COMMAND_REST, 35) == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_NEXT_TARGET, 79)
          == EMG_COMMAND_REST);
    CHECK(judge(&activation, EMG_COMMAND_CONFIRM, 736)
          == EMG_COMMAND_CONFIRM);
}

/* One entry per 50 ms hop, shaped like the measured recordings rather than
 * like tidy synthetic blocks. */
static size_t build_fixture_sequence(uint8_t *predictions, uint8_t *valid,
                                     int32_t *totals, size_t capacity)
{
    static const uint8_t classes[] = {
        EMG_COMMAND_REST, EMG_COMMAND_NEXT_TARGET,
        EMG_COMMAND_CONFIRM, EMG_COMMAND_ABORT,
    };
    uint32_t state = 0x2468acefu;
    size_t count = 0u;

#define EMIT(command, is_valid, total)                                       \
    do {                                                                     \
        if (count < capacity) {                                              \
            predictions[count] = (uint8_t)(command);                         \
            valid[count] = (uint8_t)(is_valid);                              \
            totals[count] = (int32_t)(total);                                \
            count++;                                                         \
        }                                                                    \
    } while (0)
#define REPEAT(times, command, is_valid, total)                              \
    do {                                                                     \
        for (int repeat = 0; repeat < (times); repeat++) {                   \
            EMIT((command), (is_valid), (total));                            \
        }                                                                    \
    } while (0)

    /* A gesture already under way at power-on: passes through unjudged. */
    REPEAT(6, EMG_COMMAND_NEXT_TARGET, 1, 60);
    /* First rest seeds the baseline near the measured resting 29. */
    REPEAT(10, EMG_COMMAND_REST, 1, 29);
    /* The measured defect: a preparatory ramp around 2x rest, then the
     * intended fist around 20x, then a decay that crosses the threshold on
     * the way down. */
    for (int index = 0; index < 17; index++) {
        EMIT(EMG_COMMAND_NEXT_TARGET, 1, 45 + 3 * index);
    }
    for (int index = 0; index < 10; index++) {
        EMIT(EMG_COMMAND_CONFIRM, 1, 300 + 44 * index);
    }
    REPEAT(14, EMG_COMMAND_CONFIRM, 1, 700);
    for (int index = 0; index < 6; index++) {
        EMIT(EMG_COMMAND_CONFIRM, 1, 180 - 25 * index);
    }
    /* Rest that has drifted upward: the EMA must follow. */
    REPEAT(30, EMG_COMMAND_REST, 1, 35);
    /* Dropout inside a loud gesture: passthrough, no state change. */
    REPEAT(5, EMG_COMMAND_NEXT_TARGET, 1, 600);
    EMIT(EMG_COMMAND_NEXT_TARGET, 0, 600);
    REPEAT(5, EMG_COMMAND_NEXT_TARGET, 1, 600);
    REPEAT(8, EMG_COMMAND_REST, 1, 33);
    /* A weak ABORT is suppressed like anything else; a real one passes. */
    REPEAT(8, EMG_COMMAND_ABORT, 1, 70);
    REPEAT(4, EMG_COMMAND_REST, 1, 30);
    REPEAT(8, EMG_COMMAND_ABORT, 1, 420);
    REPEAT(8, EMG_COMMAND_REST, 1, 30);

    /* Segmented pseudo-random tail: rest, sub-threshold movement, and loud
     * gestures arrive in runs, with the occasional unreadable window. */
    while (count < capacity) {
        state = state * 1664525u + 1013904223u;
        const uint32_t draw = (state >> 16) & 0xffffu;
        const uint8_t command = classes[(draw >> 8) & 0x3u];
        int32_t level;
        if (command == EMG_COMMAND_REST) {
            level = 25 + (int32_t)(draw & 0xfu);
        } else if ((draw & 0x30u) == 0u) {
            level = 45 + (int32_t)(draw & 0x3fu);
        } else {
            level = 250 + (int32_t)(draw & 0x1ffu);
        }
        const int length = 3 + (int)((draw >> 3) % 14u);
        for (int index = 0; index < length; index++) {
            state = state * 1664525u + 1013904223u;
            /* About one window in sixty-four is unreadable. */
            const uint8_t is_valid = (uint8_t)(((state >> 21) & 0x3fu) != 0u);
            EMIT(command, is_valid, level);
        }
    }

#undef REPEAT
#undef EMIT
    return count;
}

#define FIXTURE_LENGTH 1024u

static int emit_golden(const char *path)
{
    static uint8_t predictions[FIXTURE_LENGTH];
    static uint8_t valid[FIXTURE_LENGTH];
    static int32_t totals[FIXTURE_LENGTH];
    static uint8_t decisions[FIXTURE_LENGTH];
    static int32_t baselines[FIXTURE_LENGTH];
    emg_activation_t activation;
    FILE *file = NULL;
    int32_t steps = 0;
    int suppressed = 0;
    int passed = 0;

    if (!emg_activation_init(&activation, EMG_ACTIVATION_FACTOR,
                             EMG_ACTIVATION_BASELINE_SHIFT)) {
        printf("  FAIL could not init activation\n");
        return 1;
    }
    steps = (int32_t)build_fixture_sequence(predictions, valid, totals,
                                            FIXTURE_LENGTH);
    for (int32_t index = 0; index < steps; index++) {
        const emg_command_t decision = emg_activation_apply(
            &activation, (emg_command_t)predictions[index],
            valid[index] != 0u, totals[index]);
        decisions[index] = (uint8_t)decision;
        baselines[index] = emg_activation_baseline(&activation);
        if (valid[index] != 0u
            && predictions[index] != (uint8_t)EMG_COMMAND_REST
            && decision == EMG_COMMAND_REST) {
            suppressed++;
        }
        if (decision != EMG_COMMAND_REST) {
            passed++;
        }
    }
    /* A fixture that never suppresses, or never lets a gesture through,
     * would agree with any mirror while testing almost nothing. */
    if (suppressed == 0 || passed == 0) {
        printf("  FAIL fixture is hollow (suppressed=%d, passed=%d)\n",
               suppressed, passed);
        return 1;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    fwrite(&steps, sizeof(steps), 1, file);
    const uint16_t header[] = {
        EMG_ACTIVATION_FACTOR, EMG_ACTIVATION_BASELINE_SHIFT,
    };
    fwrite(header, sizeof(header[0]), sizeof(header) / sizeof(header[0]),
           file);
    fwrite(predictions, sizeof(predictions[0]), (size_t)steps, file);
    fwrite(valid, sizeof(valid[0]), (size_t)steps, file);
    fwrite(totals, sizeof(totals[0]), (size_t)steps, file);
    fwrite(decisions, sizeof(decisions[0]), (size_t)steps, file);
    fwrite(baselines, sizeof(baselines[0]), (size_t)steps, file);
    fclose(file);

    printf("  wrote %s (%d steps, %d suppressed, %d passed)\n", path,
           (int)steps, suppressed, passed);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_golden(argv[2]);
    }
    printf("test_emg_activation\n");
    test_init_rejects_configs_that_disable_protection();
    test_everything_passes_until_rest_seeds_the_baseline();
    test_low_activation_shapes_become_rest_after_seeding();
    test_threshold_scales_with_the_measured_baseline();
    test_suppressed_windows_never_feed_the_baseline();
    test_baseline_tracks_drift_without_a_deadband();
    test_rest_windows_update_the_baseline_even_when_loud();
    test_invalid_windows_change_nothing();
    test_abort_is_judged_like_every_other_class();
    test_corrupt_totals_cannot_poison_the_accumulator();
    test_candidate_constants_are_what_was_swept();
    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
