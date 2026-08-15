/* Host tests for the firmware's host-to-device receive path.
 *
 * The bytes fed in come from the same encoder the Python host mirrors byte
 * for byte (checked via fixture.bin), so passing here means the firmware
 * parses what the production sender actually produces. The corruption cases
 * are built by mutating valid packets, never by hand-assembling parallel
 * ones, so a layout change cannot silently invalidate the tests.
 */

#include "emg_rx.h"

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

static size_t encode_set(uint8_t *out, size_t out_size, uint16_t sequence,
                         uint8_t mode, uint8_t factor, uint8_t shift,
                         int32_t floor_value)
{
    emg_set_activation_t request = {0};

    request.mode = mode;
    request.factor = factor;
    request.baseline_shift = shift;
    request.threshold_floor = floor_value;
    return emg_encode_set_activation(out, out_size, sequence, &request);
}

static void test_a_valid_set_round_trips_with_its_sequence(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t wire[EMG_RX_MAX_PACKET];
    const size_t written = encode_set(wire, sizeof(wire), 0x0BEEu, 1u, 3u,
                                      4u, 110);

    emg_rx_init(&rx);
    CHECK(written == EMG_RX_MAX_PACKET);
    emg_rx_push(&rx, wire, (uint32_t)written);

    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 0x0BEEu);
    CHECK(request.mode == 1u);
    CHECK(request.factor == 3u);
    CHECK(request.baseline_shift == 4u);
    CHECK(request.threshold_floor == 110);
    CHECK(rx.accepted == 1u);
    CHECK(rx.malformed == 0u);
    CHECK(!emg_rx_poll(&rx, &request));
}

static void test_bytes_split_one_at_a_time_still_parse(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t wire[EMG_RX_MAX_PACKET];
    const size_t written = encode_set(wire, sizeof(wire), 7u, 0u, 3u, 4u,
                                      110);
    int parsed = 0;

    emg_rx_init(&rx);
    for (size_t index = 0; index < written; index++) {
        emg_rx_push(&rx, &wire[index], 1u);
        if (emg_rx_poll(&rx, &request)) {
            parsed++;
        }
    }
    CHECK(parsed == 1);
    CHECK(request.mode == 0u);
    CHECK(request.sequence == 7u);
}

static void test_corrupt_crc_resyncs_and_the_next_packet_survives(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t first[EMG_RX_MAX_PACKET];
    uint8_t second[EMG_RX_MAX_PACKET];

    (void)encode_set(first, sizeof(first), 1u, 1u, 3u, 4u, 110);
    (void)encode_set(second, sizeof(second), 2u, 1u, 3u, 4u, 200);
    first[EMG_HEADER_SIZE] ^= 0xFFu; /* corrupt the payload under the CRC */

    emg_rx_init(&rx);
    emg_rx_push(&rx, first, sizeof(first));
    emg_rx_push(&rx, second, sizeof(second));

    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 2u);
    CHECK(request.threshold_floor == 200);
    CHECK(!emg_rx_poll(&rx, &request));
    CHECK(rx.accepted == 1u);
    CHECK(rx.malformed >= 1u);
}

static void test_uplink_and_unknown_types_are_discarded(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t foreign[EMG_MAX_PACKET];
    uint8_t wire[EMG_RX_MAX_PACKET];
    const emg_intent_t intent = {EMG_COMMAND_CONFIRM, 200u, 150u, 0, 0u};
    const size_t foreign_written =
        emg_encode_intent(foreign, sizeof(foreign), 0u, 5000u, &intent);

    /* An echo of device-to-host traffic (a looped-back INTENT) must never
     * come out of the downlink parser as a configuration request. */
    emg_rx_init(&rx);
    emg_rx_push(&rx, foreign, (uint32_t)foreign_written);
    CHECK(!emg_rx_poll(&rx, &request));
    CHECK(rx.malformed >= 1u);

    (void)encode_set(wire, sizeof(wire), 9u, 1u, 3u, 4u, 110);
    emg_rx_push(&rx, wire, sizeof(wire));
    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 9u);
    CHECK(rx.accepted == 1u);
}

static void test_wrong_length_for_the_type_is_rejected_immediately(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t bad[EMG_MAX_PACKET];
    uint8_t good[EMG_RX_MAX_PACKET];
    const uint8_t short_payload[4] = {1u, 3u, 4u, 0u};
    /* A syntactically valid packet of the right type but the wrong payload
     * size. The uplink 512-byte bound would accept far more than the
     * staging area holds, so the exact-length rule is what keeps a corrupt
     * length from stalling the parser. */
    const size_t bad_written =
        emg_encode(bad, sizeof(bad), (uint8_t)EMG_TYPE_SET_ACTIVATION, 3u,
                   0u, short_payload, (uint16_t)sizeof(short_payload));

    emg_rx_init(&rx);
    emg_rx_push(&rx, bad, (uint32_t)bad_written);
    CHECK(!emg_rx_poll(&rx, &request));
    CHECK(rx.malformed >= 1u);

    (void)encode_set(good, sizeof(good), 4u, 1u, 3u, 4u, 110);
    emg_rx_push(&rx, good, sizeof(good));
    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 4u);
}

static void test_wrong_version_is_rejected_by_the_version_rule(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t wire[EMG_RX_MAX_PACKET];
    const size_t written = encode_set(wire, sizeof(wire), 5u, 1u, 3u, 4u,
                                      110);
    uint16_t crc;

    /* Change the version and re-seal the CRC so the version check, not the
     * CRC check, is what rejects it. */
    wire[2] = EMG_PROTOCOL_VERSION + 1u;
    crc = emg_crc16(&wire[2],
                    (EMG_HEADER_SIZE - 2u) + EMG_SET_ACTIVATION_PAYLOAD_SIZE);
    wire[written - 2u] = (uint8_t)(crc & 0xFFu);
    wire[written - 1u] = (uint8_t)(crc >> 8);

    emg_rx_init(&rx);
    emg_rx_push(&rx, wire, (uint32_t)written);
    CHECK(!emg_rx_poll(&rx, &request));
    CHECK(rx.malformed >= 1u);
    CHECK(rx.accepted == 0u);
}

static void test_ring_overflow_drops_bytes_and_then_recovers(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t junk[300];
    uint8_t wire[EMG_RX_MAX_PACKET];

    memset(junk, 0, sizeof(junk));
    emg_rx_init(&rx);
    emg_rx_push(&rx, junk, sizeof(junk));
    CHECK(rx.overflow_dropped > 0u);

    /* Drain the junk out; each poll moves a staging-full and discards it. */
    while (emg_rx_poll(&rx, &request)) {
        CHECK(!"junk must never parse as a request");
    }
    for (int round = 0; round < 8; round++) {
        (void)emg_rx_poll(&rx, &request);
    }

    (void)encode_set(wire, sizeof(wire), 6u, 1u, 3u, 4u, 110);
    emg_rx_push(&rx, wire, sizeof(wire));
    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 6u);
}

static void test_a_packet_torn_by_overflow_fails_crc_and_resyncs(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t junk[EMG_RX_RING_SIZE - 8u];
    uint8_t wire[EMG_RX_MAX_PACKET];

    memset(junk, 0, sizeof(junk));
    (void)encode_set(wire, sizeof(wire), 8u, 1u, 3u, 4u, 110);

    /* Leave 8 bytes of ring space so the packet is torn mid-body. */
    emg_rx_init(&rx);
    emg_rx_push(&rx, junk, sizeof(junk));
    emg_rx_push(&rx, wire, sizeof(wire));
    CHECK(rx.overflow_dropped > 0u);

    for (int round = 0; round < 12; round++) {
        CHECK(!emg_rx_poll(&rx, &request));
    }

    /* The complete resend must parse even though its head was torn off. */
    emg_rx_push(&rx, wire, sizeof(wire));
    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 8u);
}

static void test_two_requests_in_one_push_arrive_in_order(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t wire[2u * EMG_RX_MAX_PACKET];
    const size_t first = encode_set(wire, sizeof(wire), 1u, 1u, 3u, 4u, 110);
    const size_t second = encode_set(wire + first, sizeof(wire) - first, 2u,
                                     0u, 3u, 4u, 110);

    emg_rx_init(&rx);
    emg_rx_push(&rx, wire, (uint32_t)(first + second));

    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 1u && request.mode == 1u);
    CHECK(emg_rx_poll(&rx, &request));
    CHECK(request.sequence == 2u && request.mode == 0u);
    CHECK(!emg_rx_poll(&rx, &request));
}

static void test_null_arguments_are_safe(void)
{
    emg_rx_t rx;
    emg_set_activation_t request;
    uint8_t byte = 0u;

    emg_rx_init(NULL);
    emg_rx_init(&rx);
    emg_rx_push(NULL, &byte, 1u);
    emg_rx_push(&rx, NULL, 1u);
    CHECK(!emg_rx_poll(NULL, &request));
    CHECK(!emg_rx_poll(&rx, NULL));
    CHECK(!emg_rx_poll(&rx, &request));
}

int main(void)
{
    printf("test_emg_rx\n");
    test_a_valid_set_round_trips_with_its_sequence();
    test_bytes_split_one_at_a_time_still_parse();
    test_corrupt_crc_resyncs_and_the_next_packet_survives();
    test_uplink_and_unknown_types_are_discarded();
    test_wrong_length_for_the_type_is_rejected_immediately();
    test_wrong_version_is_rejected_by_the_version_rule();
    test_ring_overflow_drops_bytes_and_then_recovers();
    test_a_packet_torn_by_overflow_fails_crc_and_resyncs();
    test_two_requests_in_one_push_arrive_in_order();
    test_null_arguments_are_safe();
    if (failures == 0) {
        printf("  all checks passed\n");
        return 0;
    }
    printf("  %d check(s) failed\n", failures);
    return 1;
}
