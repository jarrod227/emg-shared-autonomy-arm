/* Host tests for the sEMG wire encoder. Built with plain gcc, no ARM
 * toolchain and no board: everything under test is pure logic.
 *
 *   make check
 *
 * `make check` also writes fixture.bin, which the Python parser tests read
 * back. The point of that file is cross-implementation agreement -- the C
 * encoder and the Python decoder are written independently from PROTOCOL.md,
 * so if they agree on these bytes the spec has no ambiguity in it.
 */

#include "emg_packet.h"

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

static uint16_t read_u16(const uint8_t *data)
{
    return (uint16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
}

static uint32_t read_u32(const uint8_t *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void test_crc_reference_vector(void)
{
    /* The published check value for CRC-16/CCITT-FALSE. If this passes, the
     * polynomial, init value, and reflection settings are all right, and any
     * independent implementation can be verified against the same constant. */
    const uint8_t input[] = "123456789";

    CHECK(emg_crc16(input, 9) == 0x29B1u);
}

static void test_header_layout(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const uint8_t payload[3] = {0xDE, 0xAD, 0xBE};
    const size_t written = emg_encode(out, sizeof(out), EMG_TYPE_INTENT,
                                      0x1234u, 0xAABBCCDDu, payload, 3u);

    CHECK(written == EMG_HEADER_SIZE + 3u + EMG_CRC_SIZE);
    CHECK(out[0] == EMG_MAGIC_0);
    CHECK(out[1] == EMG_MAGIC_1);
    CHECK(out[2] == EMG_PROTOCOL_VERSION);
    CHECK(out[3] == EMG_TYPE_INTENT);
    CHECK(read_u16(&out[4]) == 3u);
    CHECK(read_u16(&out[6]) == 0x1234u);
    CHECK(read_u32(&out[8]) == 0xAABBCCDDu);
    CHECK(memcmp(&out[12], payload, 3) == 0);

    /* The CRC must cover version..payload and exclude the magic. */
    CHECK(read_u16(&out[15]) == emg_crc16(&out[2], EMG_HEADER_SIZE - 2u + 3u));
}

static void test_rejects_oversized_and_short_buffers(void)
{
    uint8_t out[EMG_MAX_PACKET];
    uint8_t payload[EMG_MAX_PAYLOAD + 1];
    uint16_t samples[EMG_MAX_PAYLOAD];

    memset(payload, 0, sizeof(payload));
    memset(samples, 0, sizeof(samples));

    CHECK(emg_encode(out, sizeof(out), EMG_TYPE_RAW, 0u, 0u, payload,
                     (uint16_t)(EMG_MAX_PAYLOAD + 1u)) == 0u);
    /* One byte short of the full packet must be refused, not truncated. */
    CHECK(emg_encode(out, EMG_HEADER_SIZE + 3u + EMG_CRC_SIZE - 1u,
                     EMG_TYPE_INTENT, 0u, 0u, payload, 3u) == 0u);
    CHECK(emg_encode(NULL, sizeof(out), EMG_TYPE_INTENT, 0u, 0u, payload, 3u) == 0u);
    CHECK(emg_encode(out, sizeof(out), EMG_TYPE_INTENT, 0u, 0u, NULL, 3u) == 0u);
    /* The 2-byte RAW header leaves room for 255 samples, so 256 must fail. */
    CHECK(emg_encode_raw(out, sizeof(out), 0u, 0u, 0x07u, samples,
                         EMG_RAW_MAX_SAMPLES) != 0u);
    CHECK(emg_encode_raw(out, sizeof(out), 0u, 0u, 0x07u, samples,
                         EMG_RAW_MAX_SAMPLES + 1u) == 0u);
    CHECK(emg_encode_raw(out, sizeof(out), 0u, 0u, 0x07u, NULL, 4u) == 0u);
}

static void test_empty_payload_is_legal(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const size_t written = emg_encode(out, sizeof(out), EMG_TYPE_INFO, 7u, 9u,
                                      NULL, 0u);

    CHECK(written == EMG_HEADER_SIZE + EMG_CRC_SIZE);
    CHECK(read_u16(&out[4]) == 0u);
}

static void test_raw_samples_are_little_endian_and_unmodified(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const uint16_t samples[4] = {0u, 1u, 2048u, 4095u};
    const size_t written =
        emg_encode_raw(out, sizeof(out), 5u, 100u, 0x05u, samples, 4u);

    CHECK(written == EMG_HEADER_SIZE + EMG_RAW_HEADER_SIZE + 8u + EMG_CRC_SIZE);
    CHECK(read_u16(&out[4]) == EMG_RAW_HEADER_SIZE + 8u);
    CHECK(out[EMG_HEADER_SIZE] == 0x05u);
    CHECK(out[EMG_HEADER_SIZE + 1] == 0u);
    for (int index = 0; index < 4; index++) {
        CHECK(read_u16(&out[EMG_HEADER_SIZE + EMG_RAW_HEADER_SIZE + index * 2])
              == samples[index]);
    }
}

static void test_wear_mask_travels_with_its_samples(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const uint16_t samples[3] = {100u, 200u, 300u};

    /* Only channels 0 and 2 in contact: the middle electrode is off. A
     * recording has to be able to say that about the very frames it holds. */
    CHECK(emg_encode_raw(out, sizeof(out), 0u, 0u, 0x05u, samples, 3u) != 0u);
    CHECK(out[EMG_HEADER_SIZE] == 0x05u);
    CHECK(emg_encode_raw(out, sizeof(out), 0u, 0u, 0x00u, samples, 3u) != 0u);
    CHECK(out[EMG_HEADER_SIZE] == 0x00u);
}

static void test_info_and_intent_fields_round_trip(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const emg_info_t info = {0x0102u, 2000u, 3u, 12u, 32u};
    const emg_intent_t intent = {EMG_COMMAND_CONFIRM, 200u, 150u, -1, 40000u};

    CHECK(emg_encode_info(out, sizeof(out), 0u, 0u, &info) ==
          EMG_HEADER_SIZE + EMG_INFO_PAYLOAD_SIZE + EMG_CRC_SIZE);
    CHECK(read_u16(&out[12]) == 0x0102u);
    CHECK(read_u16(&out[14]) == 2000u);
    CHECK(out[16] == 3u && out[17] == 12u && out[18] == 32u && out[19] == 0u);

    CHECK(emg_encode_intent(out, sizeof(out), 0u, 0u, &intent) ==
          EMG_HEADER_SIZE + EMG_INTENT_PAYLOAD_SIZE + EMG_CRC_SIZE);
    CHECK(out[12] == EMG_COMMAND_CONFIRM);
    CHECK(out[13] == 200u && out[14] == 150u);
    CHECK((int8_t)out[15] == -1);
    CHECK(read_u16(&out[16]) == 40000u);
}

static void test_activation_state_layout(void)
{
    uint8_t out[EMG_MAX_PACKET];
    const emg_activation_state_t state = {
        1u, 3u, 4u, 1u, 110, 0x0102u,
    };

    CHECK(emg_encode_activation_state(out, sizeof(out), 11u, 7000u, &state)
          == EMG_HEADER_SIZE + EMG_ACTIVATION_STATE_PAYLOAD_SIZE
             + EMG_CRC_SIZE);
    CHECK(out[3] == EMG_TYPE_ACTIVATION_STATE);
    CHECK(read_u16(&out[4]) == EMG_ACTIVATION_STATE_PAYLOAD_SIZE);
    CHECK(out[12] == 1u);                    /* source */
    CHECK(out[13] == 3u && out[14] == 4u);   /* factor, shift */
    CHECK(out[15] == 1u);                    /* last_result */
    CHECK(read_u32(&out[16]) == 110u);       /* threshold_floor */
    CHECK(read_u16(&out[20]) == 0x0102u);    /* applied_sequence */
    CHECK(out[22] == 0u && out[23] == 0u);   /* reserved */
    CHECK(emg_encode_activation_state(out, sizeof(out), 0u, 0u, NULL) == 0u);
}

static void test_set_activation_layout_and_zero_timestamp(void)
{
    uint8_t out[EMG_MAX_PACKET];
    emg_set_activation_t request = {0};

    request.mode = 1u;
    request.factor = 3u;
    request.baseline_shift = 4u;
    request.threshold_floor = 110;
    CHECK(emg_encode_set_activation(out, sizeof(out), 5u, &request)
          == EMG_HEADER_SIZE + EMG_SET_ACTIVATION_PAYLOAD_SIZE
             + EMG_CRC_SIZE);
    CHECK(out[3] == EMG_TYPE_SET_ACTIVATION);
    /* The host does not own the device clock: timestamp is 0 by contract. */
    CHECK(read_u32(&out[8]) == 0u);
    CHECK(out[12] == 1u);
    CHECK(out[13] == 3u && out[14] == 4u);
    CHECK(out[15] == 0u);                    /* reserved */
    CHECK(read_u32(&out[16]) == 110u);
    CHECK(emg_encode_set_activation(out, sizeof(out), 0u, NULL) == 0u);
}

/* Written to fixture.bin for the Python parser tests. Deliberately includes
 * leading junk, a sequence gap, and a repeated sequence so the decoder's
 * resync and loss/duplicate accounting are exercised against bytes this
 * encoder actually produced. */
static int emit_fixture(const char *path)
{
    uint8_t out[EMG_MAX_PACKET];
    const emg_info_t info = {0x0102u, 2000u, 3u, 12u, 32u};
    const emg_intent_t intent = {EMG_COMMAND_CONFIRM, 200u, 150u, -1, 40000u};
    const uint16_t samples[6] = {0u, 1u, 2048u, 4095u, 1234u, 777u};
    const uint8_t junk[5] = {0x00, 0xA5, 0xFF, 0x5A, 0x11};
    FILE *file = fopen(path, "wb");

    if (file == NULL) {
        printf("  FAIL could not open %s\n", path);
        return 1;
    }
    fwrite(junk, 1, sizeof(junk), file);
    fwrite(out, 1, emg_encode_info(out, sizeof(out), 0u, 1000u, &info), file);
    fwrite(out, 1,
           emg_encode_raw(out, sizeof(out), 0u, 2000u, 0x07u, samples, 6u), file);
    fwrite(out, 1,
           emg_encode_raw(out, sizeof(out), 1u, 18000u, 0x07u, samples, 6u), file);
    /* Jumps 2 -> one RAW packet lost. Also drops channel 1's electrode, so
     * the decoder side has a non-trivial mask to read back. */
    fwrite(out, 1,
           emg_encode_raw(out, sizeof(out), 3u, 34000u, 0x05u, samples, 6u), file);
    fwrite(out, 1, emg_encode_intent(out, sizeof(out), 0u, 5000u, &intent), file);
    /* Repeats sequence 0 -> one INTENT duplicate. */
    fwrite(out, 1, emg_encode_intent(out, sizeof(out), 0u, 5000u, &intent), file);
    {
        /* The two 2026-08-15 additions, so the Python side can check its
         * decoder against these bytes and its own SET encoder against the
         * exact bytes this one produced. */
        const emg_activation_state_t state = {1u, 3u, 4u, 1u, 110, 0x0102u};
        emg_set_activation_t request = {0};

        request.mode = 1u;
        request.factor = 3u;
        request.baseline_shift = 4u;
        request.threshold_floor = 110;
        fwrite(out, 1,
               emg_encode_activation_state(out, sizeof(out), 0u, 6000u,
                                           &state), file);
        fwrite(out, 1,
               emg_encode_set_activation(out, sizeof(out), 5u, &request),
               file);
    }
    fclose(file);
    printf("  wrote %s\n", path);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) {
        return emit_fixture(argv[2]);
    }

    printf("test_emg_packet\n");
    test_crc_reference_vector();
    test_header_layout();
    test_rejects_oversized_and_short_buffers();
    test_empty_payload_is_legal();
    test_raw_samples_are_little_endian_and_unmodified();
    test_wear_mask_travels_with_its_samples();
    test_info_and_intent_fields_round_trip();
    test_activation_state_layout();
    test_set_activation_layout_and_zero_timestamp();

    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
