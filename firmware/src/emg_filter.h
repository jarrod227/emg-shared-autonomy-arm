/* Fixed-point IIR band-pass for sEMG, as a cascade of second-order sections.
 *
 * Cortex-M3 has no FPU, so floating point here would be software-emulated on
 * every sample of every channel. The coefficients are therefore Q29 integers
 * generated on the host by `tools/emg_filter_ref.py`, and the arithmetic uses
 * a 64-bit accumulator, which the M3 reaches with a single SMLAL.
 *
 * Self-contained rather than CMSIS-DSP, for two reasons: it must build for
 * host gcc so the fixed-point behaviour can be checked against a float
 * reference without a board, and a biquad cascade is small enough that
 * vendoring a library to get one is not a trade worth making. At 3 channels
 * and 2 kHz this costs well under 1% of the CPU either way, so if CMSIS is
 * ever wanted it is a drop-in, not a rescue.
 *
 * Coefficient convention matches scipy's `sos` rows exactly:
 *
 *     y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 *
 * so the a-terms are stored as scipy emits them and negated here. Copying
 * scipy's sign convention rather than folding it in is deliberate: a sign
 * error in a biquad produces a filter that is merely wrong, not one that
 * fails.
 */

#ifndef EMG_FILTER_H
#define EMG_FILTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Coefficients are Q29, giving a range of +/-4. The largest |a1| in a stable
 * biquad is 2, and the 20 Hz section of the shipped band-pass reaches 1.95. */
#define EMG_FILTER_COEFF_BITS 29

/* Samples are shifted up by this much inside the cascade. The 20 Hz section
 * has poles very close to the unit circle, so its state needs more resolution
 * than a raw 12-bit count carries or quantization noise dominates the output.
 * Measured against a float reference, this keeps the error at the rounding
 * quantum: max 0.50 counts, RMS 0.29. */
#define EMG_FILTER_STATE_BITS 12

#define EMG_FILTER_MAX_SECTIONS 6

typedef struct {
    int32_t b0;
    int32_t b1;
    int32_t b2;
    int32_t a1;
    int32_t a2;
} emg_biquad_coeffs_t;

typedef struct {
    int32_t x1;
    int32_t x2;
    int32_t y1;
    int32_t y2;
} emg_biquad_state_t;

typedef struct {
    emg_biquad_coeffs_t sections[EMG_FILTER_MAX_SECTIONS];
    emg_biquad_state_t state[EMG_FILTER_MAX_SECTIONS];
    uint8_t section_count;
} emg_filter_t;

/* 4th-order Butterworth band-pass, 20-450 Hz at 2000 Hz, followed by notches
 * on the 50 Hz mains fundamental and its third harmonic.
 *
 * The notches are not optional polish. Mains hum lands inside the pass band,
 * so the band-pass cannot touch it, and on a real session it was 96.6% of one
 * channel's in-band power -- the rest-to-contraction contrast was 1.5x, which
 * is no signal at all. With these notches the same recording gives 7.3% and
 * 6.4x. The second harmonic was measured and contributed almost nothing, so
 * it is omitted and the cascade stays at four sections.
 *
 * Regenerate with `python3 tools/emg_filter_ref.py --emit-c` if the rate,
 * band, or mains frequency changes -- 60 Hz regions need --mains 60. The host
 * test fails if this table and scipy disagree. */
#define EMG_FILTER_DEFAULT_SECTIONS 4
extern const emg_biquad_coeffs_t
    emg_filter_20_450_notch50_at_2000[EMG_FILTER_DEFAULT_SECTIONS];

/* Copies the sections in and clears the state. Returns false for a null
 * pointer, a zero count, or more sections than EMG_FILTER_MAX_SECTIONS. */
bool emg_filter_init(emg_filter_t *filter, const emg_biquad_coeffs_t *sections,
                     uint8_t section_count);

void emg_filter_reset(emg_filter_t *filter);

/* One sample in, one out. The return is a full int32 rather than an int16:
 * a band-pass can transiently exceed its input range, and silently clipping
 * that would corrupt the amplitude features computed downstream. */
int32_t emg_filter_step(emg_filter_t *filter, int16_t sample);

void emg_filter_block(emg_filter_t *filter, const int16_t *input,
                      int32_t *output, size_t count);

#endif /* EMG_FILTER_H */
