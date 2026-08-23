/* Fixed-point four-intent LDA inference over three channels of EMG features.
 *
 * This module is pure C with no HAL dependency.  The standardizer is folded
 * into the generated affine coefficients, so runtime inference is only
 * 4 classes x 12 int32 multiplies accumulated in int64.  Scores remain in the
 * model's Q format; classification only compares them and never divides.
 */

#ifndef EMG_CLASSIFIER_H
#define EMG_CLASSIFIER_H

#include "emg_classifier_model.h"
#include "emg_features.h"
#include "emg_packet.h"

#include <stdbool.h>
#include <stdint.h>

#define EMG_CLASSIFIER_CHANNEL_COUNT 3u
#define EMG_CLASSIFIER_FEATURES_PER_CHANNEL 4u
#define EMG_CLASSIFIER_FEATURE_COUNT 12u
/* Taken from the generated model rather than repeated. These were two
 * independent 4s that had to agree and nothing made them: a regenerated model
 * with a different class count would have been indexed with the old one,
 * reading past the coefficient rows or ignoring the last class, and the
 * compiler would have said nothing. */
#define EMG_CLASSIFIER_CLASS_COUNT EMG_CLASSIFIER_MODEL_CLASS_COUNT

_Static_assert(EMG_CLASSIFIER_MODEL_FEATURE_COUNT
                   == EMG_CLASSIFIER_FEATURE_COUNT,
               "model feature count disagrees with the classifier");

typedef struct {
    int64_t scores[EMG_CLASSIFIER_CLASS_COUNT];
    emg_command_t command;
} emg_classification_t;

/* Flat feature order is fixed by the trained model:
 * ch0(MAV,RMS,WL,ZC), ch1(MAV,RMS,WL,ZC), ch2(MAV,RMS,WL,ZC).
 * Callers must provide values produced by emg_features_compute(); the int64
 * accumulator bound is proven for that feature domain, not arbitrary int32. */
bool emg_classifier_score_flat(
    const int32_t features[EMG_CLASSIFIER_FEATURE_COUNT],
    int64_t scores[EMG_CLASSIFIER_CLASS_COUNT]);

bool emg_classifier_predict(
    const emg_features_t features[EMG_CLASSIFIER_CHANNEL_COUNT],
    emg_classification_t *result);

#endif /* EMG_CLASSIFIER_H */
