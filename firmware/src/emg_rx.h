/* Host-to-device receive path: ring buffer plus downlink packet parser.
 *
 * The USB CDC receive callback runs in interrupt context and must not parse:
 * it only copies bytes into the ring via emg_rx_push(). The main loop calls
 * emg_rx_poll(), which drains the ring into a small staging area and applies
 * the same validation and resynchronization rules PROTOCOL.md imposes on the
 * host parser — magic scan, version, type, exact length, CRC, and
 * drop-one-byte recovery on any failure.
 *
 * Single-producer single-consumer by construction: push is called only from
 * the USB interrupt, poll only from the main loop. The head and tail indices
 * are each written by exactly one side, and 32-bit aligned loads/stores are
 * atomic on Cortex-M3, so no critical section is needed.
 *
 * Overflow drops the incoming bytes (configuration traffic is sparse; the
 * buffer is generous for it) and counts them. A packet torn by such a drop
 * fails CRC and is recovered from like any other corruption.
 *
 * Pure logic with no HAL dependency, compiled into the firmware and into the
 * host tests, where the bytes fed to it come from the same encoder the
 * Python host mirrors byte for byte.
 */

#ifndef EMG_RX_H
#define EMG_RX_H

#include "emg_packet.h"

#include <stdbool.h>
#include <stdint.h>

/* Power of two so the index wrap is a mask, not a division. Sized for
 * several complete SET_ACTIVATION packets (22 bytes each) plus slack for a
 * host that stutters. */
#define EMG_RX_RING_SIZE 128u

/* The largest downlink packet the parser will assemble. Bounds the staging
 * area without trusting any length field. */
#define EMG_RX_MAX_PACKET \
    (EMG_HEADER_SIZE + EMG_SET_ACTIVATION_PAYLOAD_SIZE + EMG_CRC_SIZE)

typedef struct {
    uint8_t ring[EMG_RX_RING_SIZE];
    volatile uint32_t head; /* written by push (ISR) only */
    volatile uint32_t tail; /* written by poll (main loop) only */
    uint8_t staging[2u * EMG_RX_MAX_PACKET];
    uint32_t staged;
    /* Diagnostics, owned by their writer like the indices. */
    volatile uint32_t overflow_dropped; /* bytes lost to a full ring */
    uint32_t malformed;                 /* rejected packet candidates */
    uint32_t accepted;                  /* valid SET_ACTIVATION packets */
} emg_rx_t;

void emg_rx_init(emg_rx_t *rx);

/* Interrupt side. Copies what fits, drops and counts the rest. */
void emg_rx_push(emg_rx_t *rx, const uint8_t *data, uint32_t length);

/* Main-loop side. Returns true when one valid SET_ACTIVATION was parsed and
 * written to *request (including its wire sequence). Call repeatedly until
 * it returns false; each call returns at most one request so the caller
 * applies them strictly in arrival order. */
bool emg_rx_poll(emg_rx_t *rx, emg_set_activation_t *request);

#endif /* EMG_RX_H */
