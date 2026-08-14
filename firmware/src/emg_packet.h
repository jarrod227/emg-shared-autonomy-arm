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

/* RAW payloads open with a wear bitmask and one reserved byte, so the samples
 * that follow stay 2-byte aligned. */
#define EMG_RAW_HEADER_SIZE 2u
#define EMG_RAW_MAX_SAMPLES ((EMG_MAX_PAYLOAD - EMG_RAW_HEADER_SIZE) / 2u)

typedef enum {
    EMG_TYPE_INFO = 0x00,
    EMG_TYPE_RAW = 0x01,
    EMG_TYPE_INTENT = 0x02
} emg_packet_type_t;

typedef enum {
    EMG_COMMAND_REST = 0,
    EMG_COMMAND_NEXT_TARGET = 1,
    EMG_COMMAND_CONFIRM = 2,
    EMG_COMMAND_ABORT = 3
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

#endif /* EMG_PACKET_H */
