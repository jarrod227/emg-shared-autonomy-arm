#!/usr/bin/env python3
"""Host reference for the sEMG feature extractor in ../src/emg_features.c.

Unlike the band-pass, every operation here is integer, so this reference is
required to reproduce the firmware **exactly** rather than within a tolerance.
That makes it usable for two things the firmware cannot do on its own: trying
a definition change before committing it to C, and computing features over a
recorded dataset for training, with the guarantee that the numbers the model
learns from are the ones the MCU will produce at run time.

The four features are the classical Hudgins time-domain set, which is the
established sEMG baseline the Objective 3.5 classifier is measured against
before anything heavier is considered.
"""

import math

WINDOW = 400
HOP = 100


def mean_absolute_value(samples):
    return sum(abs(int(value)) for value in samples) // len(samples)


def root_mean_square(samples):
    """floor(sqrt(mean of squares)), matching emg_isqrt and the C's ordering.

    The mean is taken before the square root, and both steps truncate, so the
    C's integer division and bit-by-bit isqrt are reproduced step for step
    rather than approximated with a float sqrt.
    """
    square_sum = sum(int(value) * int(value) for value in samples)
    return math.isqrt(square_sum // len(samples))


def waveform_length(samples):
    return sum(abs(int(samples[index]) - int(samples[index - 1]))
               for index in range(1, len(samples)))


def zero_crossings(samples, threshold):
    """Sign changes carrying at least `threshold` counts of swing.

    Exact zeros belong to neither sign, so they never form a crossing on
    their own; the amplitude gate is what keeps a resting noise floor from
    producing a large and meaningless rate.
    """
    count = 0
    for index in range(1, len(samples)):
        previous = int(samples[index - 1])
        current = int(samples[index])
        crossed = (current > 0 > previous) or (current < 0 < previous)
        if crossed and abs(current - previous) >= threshold:
            count += 1
    return count


def compute_features(samples, threshold):
    """The four features over one full window, as a plain tuple."""
    if len(samples) != WINDOW:
        raise ValueError(f"expected a {WINDOW}-sample window, got {len(samples)}")
    return (
        mean_absolute_value(samples),
        root_mean_square(samples),
        waveform_length(samples),
        zero_crossings(samples, threshold),
    )


def sliding_features(samples, threshold, window=WINDOW, hop=HOP):
    """Every feature set the firmware would emit for this sample sequence.

    The first lands once `window` samples have arrived and one follows every
    `hop` after that; partial windows never produce features, because they
    would be averaged over fewer samples and differ in scale.
    """
    results = []
    for end in range(window, len(samples) + 1, hop):
        results.append(compute_features(samples[end - window:end], threshold))
    return results
