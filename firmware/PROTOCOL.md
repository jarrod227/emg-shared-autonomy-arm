# Cheez sEMG serial protocol v1

The wire format between the STM32F103 firmware and the host tools, over USB
CDC (`/dev/ttyACM0`). Firmware C and host Python are written independently
against this document, so it has to be precise enough that neither side needs
to read the other's source.

Status: **version 1 active**. INFO and RAW packets, including the per-channel
`wear_mask`, have been exercised against the real STM32 firmware at 2000.1 Hz.
INTENT is live as of 2026-08-14: measured at 20.0 Hz on an exact 50 000 µs
grid, with its events matching a host replay of the same samples event for
event. Increment the version byte if any existing field changes meaning or
layout.

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

The gated event plus the proportional view command, emitted once per feature
hop — 20 Hz at the 50 ms hop.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `command` | `0` REST, `1` NEXT_TARGET, `2` CONFIRM, `3` ABORT |
| 1 | 1 | `confidence` | diagnostic only, see below |
| 2 | 1 | `signal_quality` | diagnostic only, see below |
| 3 | 1 | `direction` | `int8`, `-1`, `0`, or `+1` |
| 4 | 2 | `activation` | `uint16`, `0`–`65535` maps to 0.0–1.0 |
| 6 | 2 | `reserved` | write `0` |

Payload length 8.

`command` is the **event gate's output, not the classifier's**. On every hop
where no event fires it is `REST`, so a non-`REST` value means an event fired
on that hop and never means "the current window looks like this". The two
readings differ on almost every hop of a gesture — the classifier reports the
gesture for its whole duration, the gate reports it once — so a receiver that
assumes the wrong one produces roughly forty events per gesture instead of one.

`REST` still emits a packet. A silent channel is indistinguishable from a dead
one, so the absence of intent is stated rather than implied. The Python/ROS
bridge does not publish a discrete `/assistive_intent` event for `REST`.

`timestamp_us` is derived from the count of frames that entered the DSP, so it
names the frame the decision was made on and can be aligned against the RAW
stream exactly. Frames dropped before the DSP do not advance it.

#### `confidence` and `signal_quality` are diagnostic

Both are emitted so the host can see them, and **neither may be used as a
threshold**: no mapping from either to a decision has been validated, and the
onset misclassifications that motivated the gate's hold-off were confident, not
marginal, so a confidence floor would have passed them.

- `confidence` is `min(255, (top score − runner-up) >> 16)` over the model's
  fixed-point scores. Reconstructible by the host, in the model's Q format, and
  not a probability.
- `signal_quality` is `0` when any frame in the window was missing or its
  electrode detached, and otherwise `255` minus the number of samples that
  needed clamping in that window. It reports contact and headroom, not
  certainty: a clipped window can still produce a confident wrong score.

### `0x03` ACTIVATION_STATE

The activation-threshold parameters the firmware is actually judging with,
plus how it came to hold them. Emitted on the INFO cadence (startup, then
every 2 seconds) and once immediately after every accepted **or rejected**
`SET_ACTIVATION`, so a host that just sent one never waits a full period to
learn the outcome.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `source` | `0` compile-time defaults, `1` host configuration |
| 1 | 1 | `factor` | K in effect |
| 2 | 1 | `baseline_shift` | shift in effect |
| 3 | 1 | `last_result` | `0` no SET ever received, `1` last SET accepted, `2` last SET rejected (values out of range) |
| 4 | 4 | `threshold_floor` | `int32`, floor in effect |
| 8 | 2 | `applied_sequence` | `sequence` of the last **accepted** SET; `0` before the first |
| 10 | 2 | `reserved` | write `0` |

Payload length 12.

This is a state report, not an acknowledgement. An ACK can be lost and
leaves the sender guessing whether to repeat; a periodic state is idempotent
— the host sends `SET_ACTIVATION` and watches ACTIVATION_STATE until it
reflects the request, resending if it does not. It also answers "what is
this board judging with right now" without any request, which the ROS
bridge surfaces in `/diagnostics`.

### `0x80` SET_ACTIVATION (host → device)

The first host-to-device packet, carrying the per-donning calibration for
the activation threshold. Same framing, same CRC, same resync rules as the
device-to-host direction.

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 1 | `mode` | `0` discard any host configuration and return to compile-time defaults (the value fields are ignored but must be present), `1` apply the values below |
| 1 | 1 | `factor` | K, `1`–`255` |
| 2 | 1 | `baseline_shift` | `1`–`8` |
| 3 | 1 | `reserved` | write `0` |
| 4 | 4 | `threshold_floor` | `int32`, `1` to `3*32767 - 1` |

Payload length 8.

- `mode = 0` exists so a host can *deliberately un-calibrate* the board — a
  stale calibration from a previous wearer or donning is worse than none.
  The bridge sends one of the two modes on every startup; "send nothing" is
  not a state, because an un-reset MCU would silently keep the previous
  session's RAM configuration.
- `sequence` counts per type on the host side, like every other type.
- `timestamp_us` is written as `0` and ignored: the host does not own the
  device clock, and pretending otherwise would put fabricated timestamps
  next to real ones.
- Out-of-range values reject the packet as a whole (`last_result = 2`) and
  leave the previous configuration untouched — never a partial apply.
- Configuration lives in RAM only. A reset returns to defaults; the host
  re-sends on its next startup handshake. No flash writes, no wear, and no
  board that remembers the previous wearer.

The firmware receiver obeys the same resynchronization rule as the host
parser: on any malformed candidate, discard one byte and rescan for the
magic. Bytes of unknown type in this direction are discarded the same way.
Its receive buffer is small (configuration traffic is sparse); overflow
drops bytes and counts them, and a packet torn by the drop fails CRC and
resyncs like any other corruption.

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

- **2026-08-15** — added `ACTIVATION_STATE` (`0x03`) and the first
  host-to-device packet, `SET_ACTIVATION` (`0x80`), for per-donning
  activation-threshold calibration. No version bump: `0x03` is a new
  device-to-host type old parsers already skip by the unknown-type rule, and
  `0x80` is the first use of the range this document reserved for
  host-to-device traffic from the start. No existing field changed meaning
  or layout.

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
- **No host-to-device commands beyond `SET_ACTIVATION`.** `type`
  `0x81`–`0xFF` remains reserved for future host-to-device traffic
  (proportional-control calibration is the expected next tenant).
