# Objective 3.5 — STM32 edge-EMG firmware

Firmware for the Cheez sEMG board and the host-side tools that talk to it.
This tree is **not** a colcon package: it is C built with ARM GCC through
STM32CubeIDE, so `COLCON_IGNORE` keeps `colcon build` from discovering the
CubeMX-generated `CMakeLists.txt` and trying to compile ARM sources with the
host toolchain.

The ROS 2 side stays in `src/`. Only the serial bridge that publishes
`/assistive_intent` belongs there, and it is a separate package written later;
nothing in this directory imports rclpy.

## Verified hardware

Measured on 2026-08-13. See `TODO.md` P0.1b for the same facts in roadmap form
and `docs/literature_ledger.md` (`DOC-ST-RM0008`) for the manual citations.

| | |
| --- | --- |
| Board | `EX.STM32 V1.0.3`, vendor "Cheez" |
| MCU | STM32F103C8T6, Cortex-M3, 72 MHz max, **no FPU** |
| Flash / SRAM | 64 KB / 20 KB, read from silicon (`st-info --probe`: `flash: 65536`, `sram: 20480`, `chipid: 0x410`, `STM32F1xx_MD`) |
| ADC | 2 x 12-bit, 1.17 us conversion at 72 MHz, **ADCCLK capped at 14 MHz** |
| USB | MCU's own device peripheral, ST CDC middleware — `0483:5740`, `iProduct` "Cheez sEMG", enumerates as `/dev/ttyACM0`. There is no USB-UART bridge chip on the board. |
| Debug | ST-LINK/V2 clone over 3-wire SWD |

### Pin mapping

Sensor interface columns `A0`–`A5` map to `PA0`–`PA5`, so three channels sit on
ADC1 inputs 0–2. Each column takes a module on four pins: `DO` / `AI` / `VCC` /
`GND`. The analog signal is the `AI` row.

SWD is the 4-pin header at the top-left of the board, next to the `RST` button,
silkscreened left to right: `3V3` `GND` `SWDIO` `SWCLK`. Leave `3V3`
unconnected — the board self-powers over USB-C, and a clone probe's `3V3` pin
is an output, not a voltage-sense input.

## Hard constraints

These are not preferences; they come from the silicon.

- **ADC1 scan mode + DMA1 channel 1 is the only acquisition path.** RM0008
  Rev 21 p. 227: "Only ADC1 and ADC3 have this DMA capability." Table 78 lists
  ADC1 on DMA1 channel 1 and contains no ADC2 entry. DMA2 exists only on
  high-density, XL-density and connectivity-line parts, and this is
  medium-density, so there is no ADC3 and no DMA2 either.
- **Three channels are therefore sequential, not simultaneous.** Scan mode
  converts "channel 0 to channel n" in order. At 28.5+12.5 cycles on a 12 MHz
  ADCCLK that is ~3.4 us per channel, so ch0 to ch2 skew is ~6.8 us — 0.68% of
  a sample period at 1 kHz, and 1.1 degrees of phase at 450 Hz. Irrelevant to
  per-channel amplitude features, but do not describe the result as
  "synchronized sample-and-hold".
- **Never set the ADC prescaler to /4.** At 72 MHz PCLK2 that is 18 MHz,
  over the 14 MHz limit. It compiles, it runs, and the conversions are quietly
  wrong. Use /6 (12 MHz).
- **No FPU.** The 20–450 Hz filter must be fixed-point (CMSIS-DSP q15/q31
  biquads), not float.
- **64 KB Flash is the real budget.** HAL + USB CDC middleware + CMSIS-DSP is
  roughly 30–40 KB of it. Drop HAL for LL drivers if it gets tight.

## Targets

Replacing the factory firmware is mandatory, not a preference. The factory
image samples at **499.9 Hz**, below Nyquist for the planned 20–450 Hz band, and
its output is **not raw**: values are signed and zero-centred, and 500
consecutive samples read exactly 0 at rest, so it applies DC removal plus some
squelch that cannot be characterized or reproduced.

| | Target |
| --- | --- |
| Channels | 3, raw 12-bit ADC counts |
| Sample rate | 2000 Hz |
| ADC duty | ~10 us per sweep = 2% |
| Raw data rate | 3 x 2000 x 2 B = 12 KB/s, far inside USB CDC full speed |

Pick the rate against the module's **analog** band-pass, not only the signal
band: anti-aliasing happens in the analog domain and no digital filter recovers
content that already folded down.

Expect raw counts to sit near mid-rail (~2048) rather than near zero. The
module biases the AC signal so a unipolar ADC can capture both polarities;
removing that offset is the host's or firmware's job, done explicitly.

## Open questions

Blocking or near-blocking, worth asking the vendor for a schematic and the
factory firmware source:

1. **USB D+ pull-up.** STM32F103 has no internal pull-up, so the board provides
   one externally — possibly switched by a GPIO. If it is GPIO-switched and we
   do not know which pin, custom firmware will never enumerate. This is the
   most likely cause of a first-flash failure.
2. **Is an 8 MHz HSE crystal fitted?** HSE reaches 72 MHz; HSI tops out at
   64 MHz and drifts with temperature, which matters for a stable sampling
   time base.
3. **The sEMG module's band-pass cutoffs and gain.** Sets the honest minimum
   sample rate and lets the recovered amplitudes be interpreted.

## Before the first flash

The factory image is one-shot evidence that this board works, and it may
contain a vendor USB bootloader that flashing would erase:

```bash
st-flash read cheez_factory_firmware.bin 0x8000000 0x10000
```

Keep it outside this repository.

**Do not accept a CubeIDE or CubeProgrammer prompt to upgrade the probe's
firmware.** The ST-LINK/V2 here is a clone — its serial
`303030303030303030303031` is ASCII `"000000000001"` — and clone probes are
frequently bricked by the official upgrade. Firmware V2J37S7 works as is.

## Layout

| Path | Contents |
| --- | --- |
| `PROTOCOL.md` | The wire format. Authoritative for both sides. |
| `src/` | Firmware C that has no HAL dependency, so it builds for host gcc too. |
| `test/` | Host tests for `src/`, built with plain gcc. |
| `tools/` | Host-side Python. |

The CubeMX project will land in a subdirectory of its own once the toolchain
is installed; nothing in `src/` should acquire a HAL dependency, because that
is what keeps it testable on a workstation.

## Tests

Neither suite needs the ARM toolchain or the board.

```bash
make -C firmware/test check                          # C encoder + fixture.bin
python3 -m pytest firmware/tools/test_emg_protocol.py  # Python decoder
```

Run them in that order. `make check` regenerates `fixture.bin`, a stream the
C encoder produced, and the last Python test decodes it and checks every
field. The two implementations are written from `PROTOCOL.md` without reading
each other, so that test is the evidence the spec is unambiguous — if it
fails, suspect the spec before either implementation.

The fixture deliberately contains leading junk, a RAW sequence jump, and a
repeated INTENT sequence, so resynchronization and the loss/duplicate counters
are exercised against real encoder output rather than only hand-built bytes.

## Tools

Host-side, in `tools/`. Neither needs the ARM toolchain.

| Script | Purpose |
| --- | --- |
| `emg_probe.py` | Fixed-window capture that prints a text summary: sample rate, value distribution, and a per-segment breakdown showing whether contractions register. Terminates on its own. |
| `emg_scope.py` | Live scrolling plot. Close the window to stop; a text summary is printed on exit so a run leaves evidence even if nobody was watching. |

Both need `/dev/ttyACM0` access, so the user must be in `dialout` and have
logged in again since being added.

```bash
python3 firmware/tools/emg_probe.py 10
python3 firmware/tools/emg_scope.py --channels 3 --rate 2000
```
