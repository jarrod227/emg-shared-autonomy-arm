/* Host tests for the same fixed-point LDA source linked into the firmware. */

#include "emg_classifier.h"

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

static const int32_t golden_features[][EMG_CLASSIFIER_FEATURE_COUNT] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {8, 10, 900, 4, 7, 9, 850, 4, 8, 10, 920, 5},
    {120, 150, 18000, 30, 25, 35, 5000, 12, 80, 100, 14000, 25},
    {40, 50, 7000, 15, 100, 125, 15000, 28, 35, 45, 6500, 14},
    {250, 310, 42000, 55, 120, 150, 18000, 40, 200, 260, 50000, 60},
};

static void test_null_rejected(void)
{
    int32_t features[EMG_CLASSIFIER_FEATURE_COUNT] = {0};
    int64_t scores[EMG_CLASSIFIER_CLASS_COUNT];
    emg_features_t channels[EMG_CLASSIFIER_CHANNEL_COUNT] = {0};
    emg_classification_t result;

    CHECK(!emg_classifier_score_flat(NULL, scores));
    CHECK(!emg_classifier_score_flat(features, NULL));
    CHECK(!emg_classifier_predict(NULL, &result));
    CHECK(!emg_classifier_predict(channels, NULL));
}

static void test_channel_flattening_order(void)
{
    emg_features_t channels[EMG_CLASSIFIER_CHANNEL_COUNT];
    emg_classification_t result;
    int32_t flat[EMG_CLASSIFIER_FEATURE_COUNT];
    int64_t direct[EMG_CLASSIFIER_CLASS_COUNT];

    memset(channels, 0, sizeof(channels));
    for (size_t channel = 0u; channel < EMG_CLASSIFIER_CHANNEL_COUNT; channel++) {
        const size_t offset = channel * EMG_CLASSIFIER_FEATURES_PER_CHANNEL;
        channels[channel].mean_absolute_value = (int32_t)(offset + 1u);
        channels[channel].root_mean_square = (int32_t)(offset + 2u);
        channels[channel].waveform_length = (int32_t)(offset + 3u);
        channels[channel].zero_crossings = (int32_t)(offset + 4u);
    }
    for (size_t index = 0u; index < EMG_CLASSIFIER_FEATURE_COUNT; index++) {
        flat[index] = (int32_t)(index + 1u);
    }
    CHECK(emg_classifier_predict(channels, &result));
    CHECK(emg_classifier_score_flat(flat, direct));
    for (size_t class_index = 0u;
         class_index < EMG_CLASSIFIER_CLASS_COUNT;
         class_index++) {
        CHECK(result.scores[class_index] == direct[class_index]);
    }
}

static int emit_golden(const char *path)
{
    FILE *file = fopen(path, "wb");
    const int32_t rows = (int32_t)(sizeof(golden_features)
                                   / sizeof(golden_features[0]));
    const int32_t feature_count = EMG_CLASSIFIER_FEATURE_COUNT;
    const int32_t class_count = EMG_CLASSIFIER_CLASS_COUNT;

    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    fwrite(&rows, sizeof(rows), 1, file);
    fwrite(&feature_count, sizeof(feature_count), 1, file);
    fwrite(&class_count, sizeof(class_count), 1, file);
    for (int32_t row = 0; row < rows; row++) {
        int64_t scores[EMG_CLASSIFIER_CLASS_COUNT];
        emg_command_t winner = EMG_COMMAND_REST;
        CHECK(emg_classifier_score_flat(golden_features[row], scores));
        for (int32_t class_index = 1; class_index < class_count; class_index++) {
            if (scores[class_index] > scores[winner]) {
                winner = (emg_command_t)class_index;
            }
        }
        fwrite(golden_features[row], sizeof(int32_t), feature_count, file);
        fwrite(scores, sizeof(int64_t), class_count, file);
        const uint8_t command = (uint8_t)winner;
        fwrite(&command, sizeof(command), 1, file);
    }
    fclose(file);
    printf("  wrote %s (%d rows)\n", path, rows);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_golden(argv[2]);
    }
    printf("test_emg_classifier\n");
    test_null_rejected();
    test_channel_flattening_order();
    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
