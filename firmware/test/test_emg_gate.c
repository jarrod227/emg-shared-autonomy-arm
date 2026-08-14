/* Host tests for the same event gate linked into the firmware.
 *
 * The hand-written cases below pin the individual rules. The emitted fixture
 * does the job they cannot: it runs a long sequence built from the failure
 * shapes measured on real sessions, and test_emg_gate_ref.py replays the same
 * input through the Python EventGate that the frozen counts were validated
 * with. A port that agrees on four tidy cases and diverges on a held gesture
 * interrupted by a dropout would pass the former and fail the latter.
 */

#include "emg_gate.h"

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

static emg_gate_config_t small_config(void)
{
    const emg_gate_config_t config = {
        .stable_windows = 2u,
        .rest_rearm_windows = 2u,
        .refractory_windows = 2u,
        .abort_stable_windows = 2u,
        .onset_holdoff_windows = 0u,
    };
    return config;
}

/* Returns true when an event fired, ignoring which one. */
static bool push(emg_gate_t *gate, emg_command_t prediction)
{
    emg_command_t event = EMG_COMMAND_REST;
    return emg_gate_push(gate, prediction, true, &event);
}

static void test_init_rejects_counts_that_emit_on_no_evidence(void)
{
    emg_gate_t gate;
    emg_gate_config_t config = small_config();

    CHECK(emg_gate_init(&gate, &config));
    CHECK(!emg_gate_init(NULL, &config));
    CHECK(!emg_gate_init(&gate, NULL));

    config = small_config();
    config.stable_windows = 0u;
    CHECK(!emg_gate_init(&gate, &config));
    config = small_config();
    config.rest_rearm_windows = 0u;
    CHECK(!emg_gate_init(&gate, &config));
    config = small_config();
    config.abort_stable_windows = 0u;
    CHECK(!emg_gate_init(&gate, &config));

    /* Zero refractory and zero hold-off mean "rule off", not "invalid". */
    config = small_config();
    config.refractory_windows = 0u;
    config.onset_holdoff_windows = 0u;
    CHECK(emg_gate_init(&gate, &config));
}

static void test_requires_rest_then_emits_once_for_a_held_gesture(void)
{
    emg_gate_t gate;
    const emg_gate_config_t config = small_config();
    CHECK(emg_gate_init(&gate, &config));

    /* Not armed at startup: a gesture already under way when the stream opens
     * has no observed rest behind it. */
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(gate.armed);
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(push(&gate, EMG_COMMAND_NEXT_TARGET));
    /* Held, and silent from here. */
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
}

static void test_reported_event_is_the_gesture_not_the_candidate_slot(void)
{
    emg_gate_t gate;
    const emg_gate_config_t config = small_config();
    emg_command_t event = EMG_COMMAND_REST;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!emg_gate_push(&gate, EMG_COMMAND_REST, true, &event));
    CHECK(!emg_gate_push(&gate, EMG_COMMAND_REST, true, &event));
    CHECK(!emg_gate_push(&gate, EMG_COMMAND_CONFIRM, true, &event));
    CHECK(emg_gate_push(&gate, EMG_COMMAND_CONFIRM, true, &event));
    CHECK(event == EMG_COMMAND_CONFIRM);

    /* An unrelated later push must not overwrite the caller's copy. */
    event = EMG_COMMAND_ABORT;
    CHECK(!emg_gate_push(&gate, EMG_COMMAND_REST, true, &event));
    CHECK(event == EMG_COMMAND_ABORT);
}

static void test_rearm_waits_for_rest_and_refractory(void)
{
    emg_gate_t gate;
    emg_gate_config_t config = small_config();
    config.refractory_windows = 3u;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    CHECK(push(&gate, EMG_COMMAND_CONFIRM));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!gate.armed);
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(gate.armed);
    CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    CHECK(push(&gate, EMG_COMMAND_CONFIRM));
}

static void test_abort_bypasses_arming_but_latches_until_rest(void)
{
    emg_gate_t gate;
    const emg_gate_config_t config = small_config();
    emg_command_t event = EMG_COMMAND_REST;
    CHECK(emg_gate_init(&gate, &config));

    /* No rest observed yet, and a stop must not wait on one. */
    CHECK(!emg_gate_push(&gate, EMG_COMMAND_ABORT, true, &event));
    CHECK(emg_gate_push(&gate, EMG_COMMAND_ABORT, true, &event));
    CHECK(event == EMG_COMMAND_ABORT);
    CHECK(!push(&gate, EMG_COMMAND_ABORT));
    CHECK(!push(&gate, EMG_COMMAND_ABORT));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_ABORT));
    CHECK(push(&gate, EMG_COMMAND_ABORT));
}

static void test_invalid_window_clears_evidence_and_restores_onset(void)
{
    emg_gate_t gate;
    emg_gate_config_t config = small_config();
    config.onset_holdoff_windows = 2u;
    emg_command_t event = EMG_COMMAND_REST;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));  /* hold-off */
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));  /* hold-off */
    CHECK(gate.onset_holdoff == 0u);
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));

    CHECK(!emg_gate_push(&gate, EMG_COMMAND_NEXT_TARGET, false, &event));
    CHECK(!gate.armed);
    /* A dropout is not evidence of rest, so the next contraction is an onset
     * again and must serve the hold-off from the start. */
    CHECK(gate.resting);
    CHECK(!push(&gate, EMG_COMMAND_NEXT_TARGET));
    CHECK(gate.onset_holdoff == config.onset_holdoff_windows - 1u);
}

static void test_onset_holdoff_discards_a_block_longer_than_any_stable_run(void)
{
    emg_gate_t gate;
    emg_gate_config_t config = small_config();
    config.onset_holdoff_windows = 12u;
    config.stable_windows = 5u;
    config.abort_stable_windows = 5u;
    emg_command_t event = EMG_COMMAND_REST;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(gate.armed);

    /* The measured shape: a sixteen-decision block of the wrong class at the
     * onset, then the right one. A stable run alone cannot reject this -- the
     * wrong block is longer than any threshold that would still admit the
     * correct run that follows. */
    for (int index = 0; index < 12; index++) {
        CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    }
    CHECK(gate.onset_holdoff == 0u);
    for (int index = 0; index < 4; index++) {
        CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    }
    /* Four survived the hold-off against a threshold of five: the margin that
     * carried the validation session, pinned here so a change is visible. */
    CHECK(gate.candidate_run == 4u);
    for (int index = 0; index < 4; index++) {
        CHECK(!emg_gate_push(&gate, EMG_COMMAND_ABORT, true, &event));
    }
    CHECK(emg_gate_push(&gate, EMG_COMMAND_ABORT, true, &event));
    CHECK(event == EMG_COMMAND_ABORT);
}

static void test_holdoff_restarts_only_after_rest(void)
{
    emg_gate_t gate;
    emg_gate_config_t config = small_config();
    config.onset_holdoff_windows = 3u;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    CHECK(gate.onset_holdoff == 2u);

    /* A twitch back to rest must not leave a part-spent hold-off that would
     * let the next real onset through under-filtered. */
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(gate.onset_holdoff == 0u);
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_CONFIRM));
    CHECK(gate.onset_holdoff == 2u);
}

static void test_unknown_prediction_fails_closed(void)
{
    emg_gate_t gate;
    const emg_gate_config_t config = small_config();
    emg_command_t event = EMG_COMMAND_REST;
    CHECK(emg_gate_init(&gate, &config));

    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(!push(&gate, EMG_COMMAND_REST));
    CHECK(gate.armed);
    CHECK(!emg_gate_push(&gate, (emg_command_t)7, true, &event));
    CHECK(!gate.armed);
}

static void test_default_config_is_the_validated_one(void)
{
    emg_gate_config_t config;

    memset(&config, 0, sizeof(config));
    emg_gate_default_config(&config);
    CHECK(config.stable_windows == 5u);
    CHECK(config.rest_rearm_windows == 4u);
    CHECK(config.refractory_windows == 5u);
    CHECK(config.abort_stable_windows == 1u);
    CHECK(config.onset_holdoff_windows == 12u);
}

/* One decision per 50 ms hop, built from the shapes that actually broke or
 * nearly broke the gate on measured sessions rather than from tidy blocks. */
static size_t build_fixture_sequence(uint8_t *predictions, uint8_t *valid,
                                     size_t capacity)
{
    static const uint8_t classes[] = {
        EMG_COMMAND_REST, EMG_COMMAND_NEXT_TARGET,
        EMG_COMMAND_CONFIRM, EMG_COMMAND_ABORT,
    };
    uint32_t state = 0x13579bdfu;
    size_t count = 0u;

#define EMIT(command, is_valid)                                              \
    do {                                                                     \
        if (count < capacity) {                                              \
            predictions[count] = (uint8_t)(command);                         \
            valid[count] = (uint8_t)(is_valid);                              \
            count++;                                                         \
        }                                                                    \
    } while (0)
#define REPEAT(times, command, is_valid)                                     \
    do {                                                                     \
        for (int repeat = 0; repeat < (times); repeat++) {                   \
            EMIT((command), (is_valid));                                     \
        }                                                                    \
    } while (0)

    /* Gesture blocks are sized against the frozen counts on purpose: an
     * ordinary event needs the hold-off plus a full stable run behind it, so
     * anything shorter than seventeen decisions silently tests nothing. */

    /* A gesture already under way when the stream opens: never armed. */
    REPEAT(20, EMG_COMMAND_CONFIRM, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Onset block of the wrong class, then the right one -- the ABORT shape
     * measured on the validation session. */
    REPEAT(16, EMG_COMMAND_CONFIRM, 1);
    REPEAT(10, EMG_COMMAND_ABORT, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Held gesture: one event, then silence however long it is held. */
    REPEAT(30, EMG_COMMAND_NEXT_TARGET, 1);
    /* Rest too short to re-arm, then straight into another gesture. */
    REPEAT(3, EMG_COMMAND_REST, 1);
    REPEAT(24, EMG_COMMAND_CONFIRM, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Dropout partway through: the hold-off restarts, and the gesture still
     * has enough left to fire afterwards. */
    REPEAT(13, EMG_COMMAND_NEXT_TARGET, 1);
    EMIT(EMG_COMMAND_NEXT_TARGET, 0);
    REPEAT(24, EMG_COMMAND_NEXT_TARGET, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Twitch that falls back to rest before the hold-off is spent, then a real
     * one: the second must serve a full hold-off, not the remainder. */
    REPEAT(4, EMG_COMMAND_ABORT, 1);
    REPEAT(6, EMG_COMMAND_REST, 1);
    REPEAT(20, EMG_COMMAND_ABORT, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Alternating classes: candidate runs that never reach the threshold. */
    for (int index = 0; index < 24; index++) {
        EMIT((index % 2) ? EMG_COMMAND_CONFIRM : EMG_COMMAND_NEXT_TARGET, 1);
    }
    REPEAT(8, EMG_COMMAND_REST, 1);
    /* Two gestures separated by exactly the re-arm count, so the refractory
     * boundary decides whether the second one is allowed. */
    REPEAT(24, EMG_COMMAND_CONFIRM, 1);
    REPEAT(4, EMG_COMMAND_REST, 1);
    REPEAT(24, EMG_COMMAND_NEXT_TARGET, 1);
    REPEAT(8, EMG_COMMAND_REST, 1);

    /* Then a pseudo-random tail, so the fixture also covers transitions nobody
     * thought to write down. Segments rather than independent draws: real
     * decisions arrive in runs, and per-window sampling would almost never
     * produce a run long enough to clear the hold-off, leaving the tail
     * looking thorough while exercising nothing. */
    while (count < capacity) {
        state = state * 1664525u + 1013904223u;
        const uint32_t draw = (state >> 16) & 0xffffu;
        const bool rest_segment = (draw & 0xffu) < 100u;
        const uint8_t command = rest_segment
            ? (uint8_t)EMG_COMMAND_REST
            : classes[1u + ((draw >> 8) % 3u)];
        /* Long enough to sometimes clear hold-off plus a stable run, short
         * enough to sometimes fall just short of it. */
        const int length = 4 + (int)((draw >> 3) % 22u);
        for (int index = 0; index < length; index++) {
            state = state * 1664525u + 1013904223u;
            /* About one window in forty is unreadable. */
            const uint8_t is_valid = (uint8_t)(((state >> 19) & 0x27u) != 0u);
            EMIT(command, is_valid);
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
    emg_gate_config_t config;
    emg_gate_t gate;
    FILE *file = NULL;
    int32_t event_count = 0;
    int32_t steps = 0;

    emg_gate_default_config(&config);
    if (!emg_gate_init(&gate, &config)) {
        printf("  FAIL could not init gate\n");
        return 1;
    }
    steps = (int32_t)build_fixture_sequence(predictions, valid, FIXTURE_LENGTH);

    file = fopen(path, "wb");
    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    /* Header: the counts the fixture was produced with, so a mismatch reports
     * itself as a configuration difference instead of an event difference. */
    fwrite(&steps, sizeof(steps), 1, file);
    const uint16_t header[] = {
        config.stable_windows, config.rest_rearm_windows,
        config.refractory_windows, config.abort_stable_windows,
        config.onset_holdoff_windows,
    };
    fwrite(header, sizeof(header[0]), sizeof(header) / sizeof(header[0]), file);
    fwrite(predictions, sizeof(predictions[0]), (size_t)steps, file);
    fwrite(valid, sizeof(valid[0]), (size_t)steps, file);

    /* Events are written after the input so a reader can consume the input,
     * run its own gate, and only then compare. */
    long events_at = ftell(file);
    fwrite(&event_count, sizeof(event_count), 1, file);
    for (int32_t index = 0; index < steps; index++) {
        emg_command_t event = EMG_COMMAND_REST;
        if (emg_gate_push(&gate, (emg_command_t)predictions[index],
                          valid[index] != 0u, &event)) {
            const uint8_t command = (uint8_t)event;
            fwrite(&index, sizeof(index), 1, file);
            fwrite(&command, sizeof(command), 1, file);
            event_count++;
        }
    }
    fseek(file, events_at, SEEK_SET);
    fwrite(&event_count, sizeof(event_count), 1, file);
    fclose(file);

    if (event_count == 0) {
        printf("  FAIL fixture produced no events\n");
        return 1;
    }
    printf("  wrote %s (%d steps, %d events)\n", path, steps, event_count);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_golden(argv[2]);
    }
    printf("test_emg_gate\n");
    test_init_rejects_counts_that_emit_on_no_evidence();
    test_requires_rest_then_emits_once_for_a_held_gesture();
    test_reported_event_is_the_gesture_not_the_candidate_slot();
    test_rearm_waits_for_rest_and_refractory();
    test_abort_bypasses_arming_but_latches_until_rest();
    test_invalid_window_clears_evidence_and_restores_onset();
    test_onset_holdoff_discards_a_block_longer_than_any_stable_run();
    test_holdoff_restarts_only_after_rest();
    test_unknown_prediction_fails_closed();
    test_default_config_is_the_validated_one();
    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
