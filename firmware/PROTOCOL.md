# Cheez sEMG serial protocol v1

The wire format between the STM32F103 firmware and the host tools, over USB
CDC (`/dev/ttyACM0`). Firmware C and host Python are written independently
against this document, so it has to be precise enough that neither side needs
to read the other's source.

Status: **draft**, not yet exercised against real firmware. Nothing below is
measured; it is a design. Revise the version byte if any field changes
meaning.

## Conventions

- **Little-endian** for every multi-byte field. Cortex-M3 and x86-64 are both
  little-endian, so neither side byte-swaps.
- Sizes are exact byte counts. There is no implicit padding or alignment; the
  C structs must be declared packed.
- "Frame" means one time instant across all channels. "Packet" means one
  framed transmission unit.

## Packet layout

Every packet is a 12-byte header, a variable payload, and a 2-byte CRC.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 2 | `magic` | `0xA5 0x5A`, in that byte order |
| 2 | 1 | `version` | `1` for this document |
| 3 | 1 | `type` | see packet types |
| 4 | 2 | `length` | payload bytes only, excludes header and CRC |
| 6 | 2 | `sequence` | increments per packet, wraps at 65536 |
| 8 | 4 | `timestamp_us` | microseconds since firmware boot |
| 12 | `length` | `payload` | |
| 12+`length` | 2 | `crc16` | |

- `length` must be **0 to 512**. A larger value is malformed by definition,
  which bounds the receiver's buffer without trusting the sender.
- `sequence` is counted **per packet type**, not globally. RAW and INTENT run
  at different rates, so a shared counter would make a gap ambiguous whenever
  a receiver cares about only one type.
- `timestamp_us` wraps every 2^32 us ≈ **71.6 minutes**. Both wraps are
  handled the same way: compare with modular arithmetic, never with `<`.

### CRC

CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, no input or
output reflection, no final XOR.

It covers `version` through the last payload byte — that is, everything
between `magic` and `crc16`. The magic is excluded because it is a resync
marker rather than content.

## Packet types

### `0x00` INFO

Sent once at startup and then every 2 seconds. It tells the host how to
interpret RAW packets, so the host never hard-codes a rate or channel count.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 2 | `firmware_version` | major in high byte, minor in low |
| 2 | 2 | `sample_rate_hz` | e.g. `2000` |
| 4 | 1 | `channel_count` | e.g. `3` |
| 5 | 1 | `adc_bits` | `12` |
| 6 | 1 | `frames_per_raw_packet` | e.g. `32` |
| 7 | 1 | `reserved` | write `0` |

Payload length 8.

### `0x01` RAW

Batched raw ADC counts. This is the dataset and debugging path.

Payload is `frames_per_raw_packet` frames, each `channel_count` values of
`uint16`, channel 0 first. Values are **unfiltered ADC counts, 0 to 4095**,
with no DC removal and no gating — that is the entire reason for replacing the
factory firmware, so a future firmware revision must not quietly start
processing them here.

`timestamp_us` is the capture time of the **first frame** in the packet.
Subsequent frames are at `1e6 / sample_rate_hz` microsecond spacing.

At 3 channels, 2000 Hz and 32 frames per packet: payload 192 B, packet 206 B,
62.5 packets/s, **12.9 kB/s**, 6.8% framing overhead. One frame per packet
would instead cost 233% overhead, which is why the batch exists.

### `0x02` INTENT

The classification result plus the proportional view command. Emitted once per
feature hop, so 20 Hz at the planned 50 ms hop.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `command` | `0` REST, `1` NEXT_TARGET, `2` CONFIRM, `3` ABORT |
| 1 | 1 | `confidence` | `0`–`255` maps to 0.0–1.0 |
| 2 | 1 | `signal_quality` | `0`–`255` maps to 0.0–1.0 |
| 3 | 1 | `direction` | `int8`, `-1`, `0`, or `+1` |
| 4 | 2 | `activation` | `uint16`, `0`–`65535` maps to 0.0–1.0 |
| 6 | 2 | `reserved` | write `0` |

Payload length 8.

`REST` still emits a packet. A silent channel is indistinguishable from a
dead one, so the absence of intent is stated rather than implied — the same
fail-closed rule the perception nodes already follow.

## Receiver requirements

The host parser must detect and count all four failure classes named in the
Objective 3.5 acceptance list, and must never let a bad packet reach the
consumer.

| Class | Test |
| --- | --- |
| **Malformed** | magic mismatch, `version` unknown, `type` unknown, `length` > 512, or CRC mismatch |
| **Lost** | `(sequence - last_sequence) mod 65536` > 1 for that type; the gap size is the number lost |
| **Duplicated** | `sequence == last_sequence` for that type |
| **Stale** | receipt wall-clock age exceeds the consumer's limit, or `timestamp_us` moves backwards other than by a legal wrap |

### Resynchronization

The stream is a byte stream, not a message stream, so the parser must assume
it can start mid-packet and that a corrupt length can point anywhere.

On any malformed packet, **discard one byte** from the front of the buffer and
rescan for `magic`. Do not skip `length` bytes — a corrupted length is exactly
the case being recovered from, and trusting it turns one bad packet into an
unbounded desync.

`0xA5 0x5A` can occur inside ADC payload data by chance. That is expected: the
magic narrows the search, and CRC plus the length bound are what actually
validate a candidate packet.

## Known gap: RAW carries no electrode-wear state

Found while writing the recorder. `INTENT` has a `signal_quality` byte, but
`RAW` has nothing, so a recorded dataset cannot mark which spans had a
detached electrode — and a detached electrode produces a plausible-looking
floating signal, not an obviously dead one. Training on it would be training
on noise labelled as gesture.

The board does provide the information: each sensor column has a wear-detect
line (`PA8`, `PA2`, `PA3` for columns A0–A2). Three candidate fixes, none
chosen yet:

1. Prepend a wear bitmask byte to the RAW payload. Cheapest, but changes the
   payload from pure samples, which the current spec is deliberate about.
2. Emit `INTENT` alongside `RAW` and correlate by timestamp. No format
   change, but `signal_quality` is one byte for all channels, not per channel.
3. Add a fourth packet type carrying per-channel wear at a low rate.

Decide before recording any dataset that will be used for training. Recording
first and adding the field later means the early sessions are unusable.

## What this does not do

- **No retransmission and no flow control.** The link is a local USB CDC pipe;
  loss is reported to the consumer, not repaired. A dropped RAW packet is a
  logged gap in the dataset, not an error to recover from.
- **No encryption or authentication.** It is a wired point-to-point link.
- **No host-to-device commands yet.** Reserved: `type` `0x80`–`0xFF` is the
  host-to-device range and stays unused in v1.
