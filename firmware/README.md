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
| MCU | STM32F103C8T6, Cortex-M3, 72 MHz max, **no FPU** (see the part-number note below) |
| Clock | 8 MHz HSE crystal fitted (`X2`, `PD0`/`PD1`), plus a 32.768 kHz LSE. 72 MHz and a stable sampling time base are both available. |
| Flash / SRAM | 64 KB / 20 KB, read from silicon (`st-info --probe`: `flash: 65536`, `sram: 20480`, `chipid: 0x410`, `STM32F1xx_MD`) |
| ADC | 2 x 12-bit, 1.17 us conversion at 72 MHz, **ADCCLK capped at 14 MHz** |
| USB | MCU's own device peripheral, ST CDC middleware — `0483:5740`, `iProduct` "Cheez sEMG", enumerates as `/dev/ttyACM0`. There is no USB-UART bridge chip on the board. |
| Debug | ST-LINK/V2 clone over 3-wire SWD |

### Pin mapping

Read from the vendor schematic, **not** from the board silkscreen — the
Arduino-style `A` numbering does not track the MCU port numbering, and
assuming it did produced a wrong mapping for four of the six channels.

| Sensor column | Analog (`AI`) | **ADC1 channel** | Wear detect (`DO`) |
| --- | --- | --- | --- |
| `A0` | `PA0` | **IN0** | `D2` = `PA8` |
| `A1` | `PA1` | **IN1** | `D3` = `PA2` |
| `A2` | **`PA4`** | **IN4** | `D4` = `PA3` |
| `A3` | `PA5` | IN5 | `D5` = `PB0` |
| `A4` | `PA6` | IN6 | `D6` = `PB1` |
| `A5` | `PA7` | IN7 | `D7` = `PB2` |

`PA2` and `PA3` are taken by the digital pins `D3` and `D4`, which is why the
analog run skips from `PA1` to `PA4`. **The three-channel scan sequence is
IN0, IN1, IN4.** Configuring IN0/IN1/IN2 instead would sample `PA2`, a digital
pin, and would still produce plausible-looking varying numbers.

Each column takes a module on four pins: `DO` / `AI` / `VCC` / `GND`. The
wear-detect line is a real input the protocol's `signal_quality` field should
be driven from — a detached electrode must invalidate intent rather than be
classified.

Other pins confirmed from the schematic: `SWDIO` = `PA13`, `SWCLK` = `PA14`,
`USB_DP` = `PA12`, `USB_DM` = `PA11`, and `BOOT0` carries a 10K pull-down, so
the part boots from flash normally and the USART1 ROM bootloader is reachable
by pulling `BOOT0` high.

### USB D+ pull-up is fixed, not GPIO-switched

`R13`, 10K from the 5V rail to `USB_DP`. There is no transistor, so **firmware
does not have to drive anything for the device to enumerate** — the risk that
custom firmware would simply never appear on the host is closed.

The value looks wrong against the spec's 1.5K to 3.3V, but the host's 15K
pull-down makes the two equivalent: 3.3 x 15/16.5 = 3.0 V versus
5 x 15/25 = 3.0 V, both above the 2.7 V full-speed detection threshold. The
board already enumerates as `0483:5740`, which settles it regardless.

The consequence is that the pull-up cannot be toggled to force a
re-enumeration. **After flashing, physically unplug and replug USB** if the
host still behaves as though the old firmware were running. `R14` and `R15`
(5.1K each) are the USB-C CC pull-downs and have nothing to do with this.

SWD is the 4-pin header at the top-left of the board, next to the `RST` button,
silkscreened left to right: `3V3` `GND` `SWDIO` `SWCLK`. Leave `3V3`
unconnected — the board self-powers over USB-C, and a clone probe's `3V3` pin
is an output, not a voltage-sense input.

### Part number: sources disagree

The vendor schematic labels the MCU **STM32F103C6Tx** (32 KB Flash, 10 KB
SRAM). The package marking and `st-info --probe` both say **C8** (64 KB,
20 KB), and `st-info` reads the factory-programmed size register on the die
itself. Two independent sources beat one, so the working assumption is C8 and
64 KB — but if anything inexplicable shows up once the image grows past
32 KB, come back to this line before debugging anything else.

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

The vendor schematic closed the two blocking ones: the D+ pull-up is fixed,
and an 8 MHz HSE crystal is fitted. What remains does not block starting the
firmware.

1. **The sEMG module's band-pass cutoffs and gain.** Sets the honest minimum
   sample rate, and without it the recovered amplitudes have no physical
   interpretation. The vendor supplied an Arduino library for a different
   product, so this may have to be measured rather than asked for: drive the
   module input with a swept sine and record the response.
2. **Is the USB-TTL bridge on the schematic actually populated?** If it is,
   `USART1` reaches a host serial port, and the ROM bootloader offers a
   flashing path with no probe at all. `lsusb` shows no bridge VID/PID, so it
   is probably an unpopulated footprint — this only matters as a fallback.
3. **Part number**, above: schematic says C6, silicon and marking say C8.

## CubeMX configuration

Not yet applied — the toolchain is still being installed. Each value below has
a reason attached, because several of them fail silently if set wrong.

**Select `STM32F103C8Tx`, not the `C6Tx` the vendor schematic shows.** C6
would give the linker a 32 KB / 10 KB budget when the die actually reports
64 KB / 20 KB.

| Where | Setting | Why |
| --- | --- | --- |
| System Core → SYS | Debug: **Serial Wire** | Defaults to `No Debug`, which generates code that reconfigures `PA13`/`PA14` as plain GPIO. Flash that once and ST-Link can no longer connect; recovery means pulling `BOOT0` high and erasing. Set this before anything else. |
| System Core → RCC | HSE: Crystal/Ceramic Resonator | The 8 MHz `X2` is fitted, and HSI would cap at 64 MHz with an RC oscillator's drift under an acquisition loop that needs a steady rate. |
| Clock Configuration | HSE 8 MHz → PLL ×9 → **72 MHz** | |
| Clock Configuration | USB prescaler **/1.5 → 48 MHz** | USB full speed needs exactly 48 MHz. CubeMX flags this one. |
| Clock Configuration | ADC prescaler **/6 → 12 MHz** | 14 MHz is the hard ADCCLK ceiling. `/4` gives 18 MHz, which compiles, runs, and quietly converts wrong. |
| Analog → ADC1 | Channels **IN0, IN1, IN4** | Board mapping, not IN0/IN1/IN2 — see the pin table. |
| Analog → ADC1 | Scan Conversion Mode enabled, Number of Conversions 3, Rank 1/2/3 = IN0/IN1/IN4 | |
| Analog → ADC1 | Sampling time 28.5 cycles | 28.5 + 12.5 = 41 ADC cycles = 3.4 us per channel at 12 MHz, so a 3-channel sweep is ~10 us. |
| Analog → ADC1 | External trigger: **TIM3 TRGO** | A timer trigger is what makes the sample rate exact. Free-running continuous mode would drift with whatever else the CPU is doing. |
| Timers → TIM3 | PSC **35**, ARR **999**, Trigger Event: Update Event | APB1 timers run at 72 MHz here (APB1 is 36 MHz but the timer clock doubles): 72e6 / 36 / 1000 = **2000 Hz**. |
| ADC1 → DMA Settings | **DMA1 Channel 1**, Circular, Peripheral→Memory, **Half Word** | Channel 1 is the only ADC path on this part. Half Word because conversions are 16-bit. |
| Connectivity → USB | Device (FS) | |
| Middleware → USB_DEVICE | Communication Device Class (Virtual Port Com) | |
| GPIO | `PA8`, `PA2`, `PA3` as inputs | Wear-detect lines for sensor columns A0/A1/A2. |

### Buffer sizing

Circular DMA plus the half-transfer interrupt gives double buffering for free:

```
buffer = 2 x 32 frames x 3 channels x 2 B = 384 B
DMA fills the second half -> half-transfer IRQ -> CPU packs the first half
DMA fills the first half  -> transfer-complete IRQ -> CPU packs the second
```

The CPU always touches the half the DMA is not writing, so no lock is needed
and a packet can never contain a torn frame. 32 frames is chosen to match one
RAW packet in `PROTOCOL.md`, so one interrupt produces exactly one packet.

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
| `src/` | **Firmware C — this is what gets flashed.** It has no HAL dependency, so the same files also build for host gcc. |
| `test/` | Host tests for `src/`, built with plain gcc. |
| `tools/` | Host-side Python. Never flashed. |

`src/` compiles twice from one source: ARM GCC produces the image on the chip,
host GCC produces the test binaries. That is the point of keeping it free of
HAL calls — fixed-point arithmetic and framing are where the mistakes live,
and finding them with `make check` is far faster than stepping through them
over a debug probe.

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
| `emg_protocol.py` | Decoder for the v1 wire format, with per-type sequence tracking and resynchronization. |
| `emg_record.py` | Records the stream to a raw byte log plus a JSON sidecar, or replays an existing log. |

`emg_record.py` stores **bytes, not decoded samples**. A recording of decoded
values is unrecoverable if the decoder had a bug; a byte log can be re-decoded
as often as needed, which is what `--replay` is for. It reports the sample
rate two ways — from the wall clock and from the firmware's own timestamps —
so a slow reader can be told apart from slow firmware.

```bash
python3 firmware/tools/emg_record.py --seconds 60 --out session.bin
python3 firmware/tools/emg_record.py --replay session.bin
```

Both need `/dev/ttyACM0` access, so the user must be in `dialout` and have
logged in again since being added.

```bash
python3 firmware/tools/emg_probe.py 10
python3 firmware/tools/emg_scope.py --channels 3 --rate 2000
```
