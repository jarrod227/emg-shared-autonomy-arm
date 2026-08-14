/* See emg_classifier.h. */

#include "emg_classifier.h"
#include "emg_classifier_model.h"

#include <stddef.h>

_Static_assert(EMG_CLASSIFIER_MODEL_CLASS_COUNT == EMG_CLASSIFIER_CLASS_COUNT,
               "classifier class count does not match generated model");
_Static_assert(EMG_CLASSIFIER_MODEL_FEATURE_COUNT == EMG_CLASSIFIER_FEATURE_COUNT,
               "classifier feature count does not match generated model");
_Static_assert(EMG_CLASSIFIER_MODEL_CLASS_0_COMMAND == EMG_COMMAND_REST,
               "model class 0 must map to REST");
_Static_assert(EMG_CLASSIFIER_MODEL_CLASS_1_COMMAND == EMG_COMMAND_NEXT_TARGET,
               "model class 1 must map to NEXT_TARGET");
_Static_assert(EMG_CLASSIFIER_MODEL_CLASS_2_COMMAND == EMG_COMMAND_CONFIRM,
               "model class 2 must map to CONFIRM");
_Static_assert(EMG_CLASSIFIER_MODEL_CLASS_3_COMMAND == EMG_COMMAND_ABORT,
               "model class 3 must map to ABORT");

bool emg_classifier_score_flat(
    const int32_t features[EMG_CLASSIFIER_FEATURE_COUNT],
    int64_t scores[EMG_CLASSIFIER_CLASS_COUNT])
{
    if (features == NULL || scores == NULL) {
        return false;
    }
    for (size_t class_index = 0u;
         class_index < EMG_CLASSIFIER_CLASS_COUNT;
         class_index++) {
        int64_t score = emg_classifier_model_intercept[class_index];
        for (size_t feature_index = 0u;
             feature_index < EMG_CLASSIFIER_FEATURE_COUNT;
             feature_index++) {
            score += (int64_t)features[feature_index]
                   * emg_classifier_model_weights[class_index][feature_index];
        }
        scores[class_index] = score;
    }
    return true;
}

bool emg_classifier_predict(
    const emg_features_t features[EMG_CLASSIFIER_CHANNEL_COUNT],
    emg_classification_t *result)
{
    int32_t flat[EMG_CLASSIFIER_FEATURE_COUNT];
    size_t position = 0u;

    if (features == NULL || result == NULL) {
        return false;
    }
    for (size_t channel = 0u; channel < EMG_CLASSIFIER_CHANNEL_COUNT; channel++) {
        flat[position++] = features[channel].mean_absolute_value;
        flat[position++] = features[channel].root_mean_square;
        flat[position++] = features[channel].waveform_length;
        flat[position++] = features[channel].zero_crossings;
    }
    if (!emg_classifier_score_flat(flat, result->scores)) {
        return false;
    }

    size_t winner = 0u;
    for (size_t class_index = 1u;
         class_index < EMG_CLASSIFIER_CLASS_COUNT;
         class_index++) {
        if (result->scores[class_index] > result->scores[winner]) {
            winner = class_index;
        }
    }
    result->command =
        (emg_command_t)emg_classifier_model_commands[winner];
    return true;
}
