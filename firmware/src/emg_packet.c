/* See emg_packet.h and PROTOCOL.md. */

#include "emg_packet.h"

#include <string.h>

/* Every multi-byte field is written byte by byte rather than memcpy'd from a
 * struct. That keeps the layout independent of the compiler's alignment and
 * of the target's endianness, and the cost is irrelevant: a full RAW packet
 * is 96 values at 62.5 packets/s. */
static void put_u16(uint8_t *out, uint16_t value)
{
    out[0] = (uint8_t)(value & 0xFFu);
    out[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static void put_u32(uint8_t *out, uint32_t value)
{
    out[0] = (uint8_t)(value & 0xFFu);
    out[1] = (uint8_t)((value >> 8) & 0xFFu);
    out[2] = (uint8_t)((value >> 16) & 0xFFu);
    out[3] = (uint8_t)((value >> 24) & 0xFFu);
}

uint16_t emg_crc16(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xFFFFu;

    if (data == NULL) {
        return crc;
    }
    for (size_t index = 0; index < length; index++) {
        crc ^= (uint16_t)((uint16_t)data[index] << 8);
        for (int bit = 0; bit < 8; bit++) {
            if ((crc & 0x8000u) != 0u) {
                crc = (uint16_t)((uint16_t)(crc << 1) ^ 0x1021u);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

size_t emg_encode(uint8_t *out, size_t out_size, uint8_t type,
                  uint16_t sequence, uint32_t timestamp_us,
                  const uint8_t *payload, uint16_t payload_length)
{
    if (out == NULL || payload_length > EMG_MAX_PAYLOAD) {
        return 0u;
    }
    if (payload_length > 0u && payload == NULL) {
        return 0u;
    }

    const size_t total = EMG_HEADER_SIZE + (size_t)payload_length + EMG_CRC_SIZE;
    if (out_size < total) {
        return 0u;
    }

    out[0] = EMG_MAGIC_0;
    out[1] = EMG_MAGIC_1;
    out[2] = (uint8_t)EMG_PROTOCOL_VERSION;
    out[3] = type;
    put_u16(&out[4], payload_length);
    put_u16(&out[6], sequence);
    put_u32(&out[8], timestamp_us);
    if (payload_length > 0u) {
        memcpy(&out[EMG_HEADER_SIZE], payload, (size_t)payload_length);
    }

    /* The CRC covers `version` through the last payload byte. The magic is
     * excluded because it is a resync marker the receiver scans for, not
     * content it needs protected. */
    const uint16_t crc = emg_crc16(&out[2],
                                   (EMG_HEADER_SIZE - 2u) + (size_t)payload_length);
    put_u16(&out[EMG_HEADER_SIZE + payload_length], crc);
    return total;
}

size_t emg_encode_info(uint8_t *out, size_t out_size, uint16_t sequence,
                       uint32_t timestamp_us, const emg_info_t *info)
{
    uint8_t payload[EMG_INFO_PAYLOAD_SIZE];

    if (info == NULL) {
        return 0u;
    }
    put_u16(&payload[0], info->firmware_version);
    put_u16(&payload[2], info->sample_rate_hz);
    payload[4] = info->channel_count;
    payload[5] = info->adc_bits;
    payload[6] = info->frames_per_raw_packet;
    payload[7] = 0u;
    return emg_encode(out, out_size, (uint8_t)EMG_TYPE_INFO, sequence,
                      timestamp_us, payload, EMG_INFO_PAYLOAD_SIZE);
}

size_t emg_encode_raw(uint8_t *out, size_t out_size, uint16_t sequence,
                      uint32_t timestamp_us, uint8_t wear_mask,
                      const uint16_t *samples, uint16_t sample_count)
{
    if (out == NULL || samples == NULL) {
        return 0u;
    }
    if (sample_count > EMG_RAW_MAX_SAMPLES) {
        return 0u;
    }

    const uint16_t payload_length =
        (uint16_t)(EMG_RAW_HEADER_SIZE + (size_t)sample_count * 2u);
    const size_t total = EMG_HEADER_SIZE + (size_t)payload_length + EMG_CRC_SIZE;
    if (out_size < total) {
        return 0u;
    }

    /* Serialize straight into the output buffer: a separate staging array
     * would be 512 bytes of stack on a part with 20 KB of SRAM. */
    out[0] = EMG_MAGIC_0;
    out[1] = EMG_MAGIC_1;
    out[2] = (uint8_t)EMG_PROTOCOL_VERSION;
    out[3] = (uint8_t)EMG_TYPE_RAW;
    put_u16(&out[4], payload_length);
    put_u16(&out[6], sequence);
    put_u32(&out[8], timestamp_us);
    out[EMG_HEADER_SIZE] = wear_mask;
    out[EMG_HEADER_SIZE + 1u] = 0u;
    for (uint16_t index = 0; index < sample_count; index++) {
        put_u16(&out[EMG_HEADER_SIZE + EMG_RAW_HEADER_SIZE
                     + (size_t)index * 2u],
                samples[index]);
    }
    const uint16_t crc = emg_crc16(&out[2],
                                   (EMG_HEADER_SIZE - 2u) + (size_t)payload_length);
    put_u16(&out[EMG_HEADER_SIZE + payload_length], crc);
    return total;
}

size_t emg_encode_intent(uint8_t *out, size_t out_size, uint16_t sequence,
                         uint32_t timestamp_us, const emg_intent_t *intent)
{
    uint8_t payload[EMG_INTENT_PAYLOAD_SIZE];

    if (intent == NULL) {
        return 0u;
    }
    payload[0] = intent->command;
    payload[1] = intent->confidence;
    payload[2] = intent->signal_quality;
    payload[3] = (uint8_t)intent->direction;
    put_u16(&payload[4], intent->activation);
    payload[6] = 0u;
    payload[7] = 0u;
    return emg_encode(out, out_size, (uint8_t)EMG_TYPE_INTENT, sequence,
                      timestamp_us, payload, EMG_INTENT_PAYLOAD_SIZE);
}
