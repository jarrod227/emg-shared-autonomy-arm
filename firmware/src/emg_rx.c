/* See emg_rx.h. */

#include "emg_rx.h"

#include <stddef.h>
#include <string.h>

void emg_rx_init(emg_rx_t *rx)
{
    if (rx == NULL) {
        return;
    }
    memset(rx, 0, sizeof(*rx));
}

void emg_rx_push(emg_rx_t *rx, const uint8_t *data, uint32_t length)
{
    if (rx == NULL || data == NULL) {
        return;
    }
    /* Read tail once: the consumer may advance it concurrently, and a stale
     * value only makes this side conservative (it drops a byte it could
     * have kept, never overwrites one it should not). */
    const uint32_t tail = rx->tail;
    uint32_t head = rx->head;
    for (uint32_t index = 0; index < length; index++) {
        if (head - tail >= EMG_RX_RING_SIZE) {
            rx->overflow_dropped += length - index;
            break;
        }
        rx->ring[head % EMG_RX_RING_SIZE] = data[index];
        head++;
    }
    /* Publish after the bytes are in place. */
    rx->head = head;
}

/* Move everything available out of the ring into the linear staging area.
 * The staging area is bounded, so a flood simply waits in the ring (and
 * overflows there); the parser below never trusts a wire length to index
 * beyond what was actually staged. */
static void drain_ring(emg_rx_t *rx)
{
    const uint32_t head = rx->head; /* read once; ISR may advance it */
    uint32_t tail = rx->tail;
    while (tail != head && rx->staged < sizeof(rx->staging)) {
        rx->staging[rx->staged++] = rx->ring[tail % EMG_RX_RING_SIZE];
        tail++;
    }
    rx->tail = tail;
}

static void discard_staged(emg_rx_t *rx, uint32_t count)
{
    memmove(rx->staging, rx->staging + count, rx->staged - count);
    rx->staged -= count;
}

/* Drop one byte and rescan, per PROTOCOL.md: a corrupted length is exactly
 * the case being recovered from, so it is never trusted as a skip count. */
static void resync(emg_rx_t *rx)
{
    rx->malformed++;
    discard_staged(rx, 1u);
}

static uint16_t get_u16(const uint8_t *in)
{
    return (uint16_t)((uint16_t)in[0] | ((uint16_t)in[1] << 8));
}

static uint32_t get_u32(const uint8_t *in)
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8)
           | ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

bool emg_rx_poll(emg_rx_t *rx, emg_set_activation_t *request)
{
    if (rx == NULL || request == NULL) {
        return false;
    }
    drain_ring(rx);

    while (rx->staged > 0u) {
        /* Scan for the magic; everything before it can never start a
         * packet. Keep a trailing first-magic-byte in case the pair
         * straddles this drain and the next. */
        uint32_t start = 0u;
        while (start + 1u < rx->staged
               && !(rx->staging[start] == EMG_MAGIC_0
                    && rx->staging[start + 1u] == EMG_MAGIC_1)) {
            start++;
        }
        if (start + 1u >= rx->staged) {
            const bool keep_last = rx->staged > 0u
                && rx->staging[rx->staged - 1u] == EMG_MAGIC_0;
            const uint32_t drop = keep_last ? rx->staged - 1u : rx->staged;
            if (drop > 0u) {
                discard_staged(rx, drop);
            }
            return false;
        }
        if (start > 0u) {
            discard_staged(rx, start);
        }
        if (rx->staged < EMG_HEADER_SIZE) {
            return false;
        }

        const uint8_t version = rx->staging[2];
        const uint8_t type = rx->staging[3];
        const uint16_t length = get_u16(&rx->staging[4]);
        if (version != EMG_PROTOCOL_VERSION) {
            resync(rx);
            continue;
        }
        /* The firmware understands exactly one downlink type, and knowing
         * its exact length is what lets a bounded staging area reject a
         * corrupt length immediately instead of waiting for bytes that a
         * 512-byte uplink bound would permit but the buffer cannot hold. */
        if (type != (uint8_t)EMG_TYPE_SET_ACTIVATION) {
            resync(rx);
            continue;
        }
        if (length != EMG_SET_ACTIVATION_PAYLOAD_SIZE) {
            resync(rx);
            continue;
        }

        const uint32_t total = EMG_HEADER_SIZE + (uint32_t)length
                               + EMG_CRC_SIZE;
        if (rx->staged < total) {
            return false;
        }

        const uint16_t expected = get_u16(&rx->staging[total - EMG_CRC_SIZE]);
        if (emg_crc16(&rx->staging[2],
                      (EMG_HEADER_SIZE - 2u) + (size_t)length) != expected) {
            resync(rx);
            continue;
        }

        request->sequence = get_u16(&rx->staging[6]);
        request->mode = rx->staging[EMG_HEADER_SIZE + 0u];
        request->factor = rx->staging[EMG_HEADER_SIZE + 1u];
        request->baseline_shift = rx->staging[EMG_HEADER_SIZE + 2u];
        request->threshold_floor =
            (int32_t)get_u32(&rx->staging[EMG_HEADER_SIZE + 4u]);
        request->reference_left =
            (int32_t)get_u32(&rx->staging[EMG_HEADER_SIZE + 8u]);
        request->reference_right =
            (int32_t)get_u32(&rx->staging[EMG_HEADER_SIZE + 12u]);
        discard_staged(rx, total);
        rx->accepted++;
        return true;
    }
    return false;
}
