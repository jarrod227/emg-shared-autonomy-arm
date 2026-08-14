# Cheez sEMG serial protocol v1

The wire format between the STM32F103 firmware and the host tools, over USB
CDC (`/dev/ttyACM0`). Firmware C and host Python are written independently
against this document, so it has to be precise enough that neither side needs
to read the other's source.

Status: **version 1 active for raw data collection**. INFO and RAW packets,
including the per-channel `wear_mask`, have been exercised against the real
STM32 firmware at 2000.1 Hz. The INTENT layout and independent C/Python
encoders/decoders are host-tested, but live firmware does not emit INTENT until
a trained classifier is integrated. Increment the version byte if any existing
field changes meaning or layout.

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

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `wear_mask` | bit *N* set means channel *N*'s electrode is in contact |
| 1 | 1 | `reserved` | write `0`; present so the samples stay 2-byte aligned |
| 2 | rest | `samples` | `frames_per_raw_packet` frames of `channel_count` `uint16` values, channel 0 first |

Values are **unfiltered ADC counts, 0 to 4095**, with no DC removal and no
gating — that is the entire reason for replacing the factory firmware, so a
future firmware revision must not quietly start processing them here.

`wear_mask` reports the board's per-channel wear-detect lines (`PA8`, `PA2`,
`PA3` for columns A0–A2). It rides in the same packet as the samples it
describes, so the two can never be misaligned by a lost packet — which is why
this is a payload field rather than a separate packet type or a
timestamp-correlated `INTENT` field. Bits above `channel_count - 1` are zero.

A cleared bit means "not in contact", and an unplugged column reads cleared
because those pins are pulled down. That is deliberate: a detached electrode
floats and produces plausible-looking signal, so a recording that cannot mark
those spans would train a classifier on noise labelled as gesture.

`timestamp_us` is the capture time of the **first frame** in the packet.
Subsequent frames are at `1e6 / sample_rate_hz` microsecond spacing.

At 3 channels, 2000 Hz and 32 frames per packet: payload 194 B, packet 208 B,
62.5 packets/s, **13.0 kB/s**, 7.7% framing overhead. One frame per packet
would instead cost over 200% overhead, which is why the batch exists.

### `0x02` INTENT

The classification result plus the proportional view command. Once live
inference is connected, it will be emitted once per feature hop, so 20 Hz at
the planned 50 ms hop.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `command` | `0` REST, `1` NEXT_TARGET, `2` CONFIRM, `3` ABORT |
| 1 | 1 | `confidence` | `0`–`255` maps to 0.0–1.0 |
| 2 | 1 | `signal_quality` | `0`–`255` maps to 0.0–1.0 |
| 3 | 1 | `direction` | `int8`, `-1`, `0`, or `+1` |
| 4 | 2 | `activation` | `uint16`, `0`–`65535` maps to 0.0–1.0 |
| 6 | 2 | `reserved` | write `0` |

Payload length 8.

Once INTENT is live, `REST` still emits a packet. A silent channel is
indistinguishable from a dead one, so the absence of intent is stated rather
than implied. The Python/ROS bridge does not publish a discrete
`/assistive_intent` event for `REST`.

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

## Amendment history

Version 1 now has real INFO/RAW recordings. Any incompatible change to an
existing packet layout or field meaning requires a version bump; adding code
that emits the already-specified INTENT packet does not.

- **2026-08-14** — exercised INFO and RAW against the real STM32 acquisition
  firmware at 2000.1 Hz with zero lost, malformed, or duplicated packets in a
  15-second run. INTENT remains specified and host-tested but not emitted.

- **2026-08-13** — added `wear_mask` and `reserved` to the `RAW` payload.
  Originally `RAW` carried only samples and the wear state had no home, which
  would have made recorded spans with a detached electrode indistinguishable
  from real signal. Putting the mask in the same packet as the samples was
  chosen over a timestamp-correlated `INTENT` field (loses alignment when a
  packet is dropped, and `signal_quality` is one byte for all channels) and
  over a fourth packet type (splits one fact across two packets that can
  desynchronize).

## What this does not do

- **No retransmission and no flow control.** The link is a local USB CDC pipe;
  loss is reported to the consumer, not repaired. A dropped RAW packet is a
  logged gap in the dataset, not an error to recover from.
- **No encryption or authentication.** It is a wired point-to-point link.
- **No host-to-device commands yet.** Reserved: `type` `0x80`–`0xFF` is the
  host-to-device range and stays unused in v1.
