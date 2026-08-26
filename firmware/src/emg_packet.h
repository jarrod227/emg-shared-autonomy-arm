/* Wire encoder for the Cheez sEMG serial protocol v1.
 *
 * Pure logic: no HAL, no CMSIS, no dynamic allocation, no I/O. It builds
 * bytes into a caller-supplied buffer so the same source compiles for the
 * STM32 target and for host gcc, and so the fiddly parts -- CRC, byte order,
 * length bounds -- can be tested on a workstation instead of on the MCU.
 *
 * See PROTOCOL.md for the authoritative field layout.
 */

#ifndef EMG_PACKET_H
#define EMG_PACKET_H

#include <stddef.h>
#include <stdint.h>

#define EMG_MAGIC_0 0xA5u
#define EMG_MAGIC_1 0x5Au
#define EMG_PROTOCOL_VERSION 1u

#define EMG_HEADER_SIZE 12u
#define EMG_CRC_SIZE 2u
#define EMG_MAX_PAYLOAD 512u
#define EMG_MAX_PACKET (EMG_HEADER_SIZE + EMG_MAX_PAYLOAD + EMG_CRC_SIZE)

#define EMG_INFO_PAYLOAD_SIZE 8u
#define EMG_INTENT_PAYLOAD_SIZE 8u
#define EMG_ACTIVATION_STATE_PAYLOAD_SIZE 20u
#define EMG_SET_ACTIVATION_PAYLOAD_SIZE 16u

/* RAW payloads open with a wear bitmask and one reserved byte, so the samples
 * that follow stay 2-byte aligned. */
#define EMG_RAW_HEADER_SIZE 2u
#define EMG_RAW_MAX_SAMPLES ((EMG_MAX_PAYLOAD - EMG_RAW_HEADER_SIZE) / 2u)

typedef enum {
    EMG_TYPE_INFO = 0x00,
    EMG_TYPE_RAW = 0x01,
    EMG_TYPE_INTENT = 0x02,
    EMG_TYPE_ACTIVATION_STATE = 0x03,
    /* 0x80-0xFF is the host-to-device range. */
    EMG_TYPE_SET_ACTIVATION = 0x80
} emg_packet_type_t;

typedef enum {
    EMG_ACTIVATION_SOURCE_DEFAULTS = 0,
    EMG_ACTIVATION_SOURCE_HOST = 1
} emg_activation_source_t;

typedef enum {
    EMG_SET_RESULT_NONE = 0,
    EMG_SET_RESULT_ACCEPTED = 1,
    EMG_SET_RESULT_REJECTED = 2
} emg_set_result_t;

typedef enum {
    EMG_SET_MODE_DEFAULTS = 0,
    EMG_SET_MODE_APPLY = 1
} emg_set_mode_t;

typedef enum {
    EMG_COMMAND_REST = 0,
    EMG_COMMAND_NEXT_TARGET = 1,
    EMG_COMMAND_CONFIRM = 2,
    EMG_COMMAND_ABORT = 3,
    /* Classifier class only, never an event. It steers the proportional view
     * channel and is rewritten to REST before the gate sees it, so it cannot
     * reach the INTENT `command` field and no receiver has to learn a fifth
     * command. Giving it one would give it a discrete meaning the design
     * deliberately does not want: while a search sweeps, a gesture is a
     * direction and nothing else. */
    EMG_COMMAND_ULNAR = 4
} emg_command_t;

typedef struct {
    uint16_t firmware_version;
    uint16_t sample_rate_hz;
    uint8_t channel_count;
    uint8_t adc_bits;
    uint8_t frames_per_raw_packet;
} emg_info_t;

typedef struct {
    uint8_t command;
    uint8_t confidence;
    uint8_t signal_quality;
    int8_t direction;
    uint16_t activation;
} emg_intent_t;

/* What the firmware is judging with right now, and how it got there. A state
 * report rather than an ACK: an ACK can be lost and leaves the sender
 * guessing, a periodic state is idempotent. */
typedef struct {
    uint8_t source;         /* emg_activation_source_t */
    uint8_t factor;
    uint8_t baseline_shift;
    uint8_t last_result;    /* emg_set_result_t */
    int32_t threshold_floor;
    uint16_t applied_sequence; /* sequence of the last accepted SET */
    /* The EMA rest baseline the board is judging with right now, saturated
     * into 16 bits. It occupies bytes the layout already reserved, so the
     * payload length does not change.
     *
     * Reported because threshold_floor is only half of what the board
     * actually uses: it judges on max(factor * baseline, threshold_floor),
     * and the baseline moves during a session while the floor does not. A
     * night was lost to gestures that fired during calibration and produced
     * nothing an hour later, with every reported number unchanged, because
     * the half that moved was the half nobody could see. */
    uint16_t baseline;
    /* Full-deflection levels for the two steering gestures, in the same
     * units as threshold_floor. Reported, not just accepted, because the
     * host confirms a calibration by watching this reflect what it sent;
     * a field it cannot see back is a field it cannot verify was applied.
     * Zero means none was supplied and the compile-time fallback is in use. */
    int32_t reference_left;
    int32_t reference_right;
} emg_activation_state_t;

/* One decoded host-to-device SET_ACTIVATION request. Range checking is the
 * applier's job, not the decoder's: a rejected request must still be
 * reportable as `last_result = 2`, which requires it to reach the caller. */
typedef struct {
    uint8_t mode;           /* emg_set_mode_t */
    uint8_t factor;
    uint8_t baseline_shift;
    int32_t threshold_floor;
    /* Measured per donning; see emg_view.h for why they cannot be derived
     * from threshold_floor. Zero for either one selects the compile-time
     * fallback for that direction, which is what SET_MODE_DEFAULTS sends. */
    int32_t reference_left;
    int32_t reference_right;
    uint16_t sequence;      /* wire sequence, echoed in applied_sequence */
} emg_set_activation_t;

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR. */
uint16_t emg_crc16(const uint8_t *data, size_t length);

/* Frame an arbitrary payload. Returns bytes written, or 0 if the payload is
 * over EMG_MAX_PAYLOAD or the output buffer is too small. Never writes past
 * out_size. */
size_t emg_encode(uint8_t *out, size_t out_size, uint8_t type,
                  uint16_t sequence, uint32_t timestamp_us,
                  const uint8_t *payload, uint16_t payload_length);

size_t emg_encode_info(uint8_t *out, size_t out_size, uint16_t sequence,
                       uint32_t timestamp_us, const emg_info_t *info);

/* sample_count is frames * channels, channel 0 first within each frame.
 * Values are raw ADC counts and are transmitted unmodified: the protocol
 * contract is that RAW carries no DC removal, filtering, or gating.
 *
 * wear_mask carries the per-channel electrode-contact lines, bit N for
 * channel N. It travels with the samples it describes rather than in a
 * separate packet so a dropped packet can never misalign the two. A cleared
 * bit means no contact, which is also what an unplugged column reads. */
size_t emg_encode_raw(uint8_t *out, size_t out_size, uint16_t sequence,
                      uint32_t timestamp_us, uint8_t wear_mask,
                      const uint16_t *samples, uint16_t sample_count);

size_t emg_encode_intent(uint8_t *out, size_t out_size, uint16_t sequence,
                         uint32_t timestamp_us, const emg_intent_t *intent);

size_t emg_encode_activation_state(uint8_t *out, size_t out_size,
                                   uint16_t sequence, uint32_t timestamp_us,
                                   const emg_activation_state_t *state);

/* Host-to-device encoder. On the MCU this exists for the host-side tests
 * (they build the bytes the firmware receiver must parse); the production
 * sender is the Python host. */
size_t emg_encode_set_activation(uint8_t *out, size_t out_size,
                                 uint16_t sequence,
                                 const emg_set_activation_t *request);

#endif /* EMG_PACKET_H */
