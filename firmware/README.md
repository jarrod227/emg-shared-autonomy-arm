# Objective 3.5 — STM32 edge-EMG firmware

Firmware for the Cheez sEMG board and the host-side tools that talk to it.

This file states what is true about the hardware and how it is configured.
How each fact was established — and what was believed beforehand — is in
`../docs/objective35_bringup_log.md`, including the errors that would have
produced plausible data rather than an error message.
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

Each column takes a module on four pins: `DO` / `AI` / `VCC` / `GND`. RAW
packets already carry the three per-channel `wear_mask` bits. A future INTENT
packet will derive its aggregate `signal_quality` from those inputs; the
electrical active-high polarity still requires an unplug test before detached
electrodes can be claimed to invalidate intent fail-closed.

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
  ADCCLK that is ~3.4 us per channel, so ch0 to ch2 skew is ~6.8 us — 1.36% of
  the current 2 kHz sample period, and 1.1 degrees of phase at 450 Hz. Irrelevant to
  per-channel amplitude features, but do not describe the result as
  "synchronized sample-and-hold".
- **Never set the ADC prescaler to /4.** At 72 MHz PCLK2 that is 18 MHz,
  over the 14 MHz limit. It compiles, it runs, and the conversions are quietly
  wrong. Use /6 (12 MHz).
- **No FPU.** The 20–450 Hz filter must be fixed-point (CMSIS-DSP q15/q31
  biquads), not float.
- **Flash turned out not to be the constraint.** Measured on the first build,
  with HAL, the USB CDC middleware, and the shared sources linked but no
  application logic yet: **18 700 B at `-Os`** (28.5% of 64 KB) against
  39 304 B at `-O0 -g3`. An earlier 30–40 KB estimate was reading the debug
  figure. Develop in Debug and expect it to be tight there; the shipping
  build is Release.
- **RAM is the tighter one.** 7 704 B (37.6%) before any of the per-channel
  state exists, and identical in both build types since static allocation
  does not depend on optimization. Adding three channels of filter and
  feature state (2 880 B), the DMA double buffer (384 B), and one transmit
  buffer (206 B) projects to roughly **11.2 KB, or 55%**.

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

## Building

The CubeMX project lives in `cheez_emg/`. The toolchain is Ubuntu's, not ST's
STM32CubeCLT: `gcc-arm-none-eabi`, `ninja-build`, and `gdb-multiarch` from apt,
with `st-flash` and `st-util` from `stlink-tools` for flashing and as a GDB
server. That covers build, flash, and debug without another ST download; the
cost is that the VS Code extension's one-click buttons are not wired to it.

```bash
cd firmware/cheez_emg
cmake --preset Debug          # or Release
cmake --build build/Debug
```

`CMakeLists.txt` pulls `../src/*.c` into the firmware image, so the framing,
filter, and feature sources are compiled by ARM GCC here and by host GCC under
`../test` — one set of files, two compilers.

## CubeMX configuration

Applied and verified against the generated sources. Each value below has a
reason attached, because several of them fail silently if set wrong.

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

`src/` compiles twice from one source: ARM GCC links it into
`cheez_emg`, while host GCC produces the test binaries. The CubeMX/HAL project
is now in `cheez_emg/`; the shared packet, filter, and feature sources remain
HAL-free so their exact fixed-point and framing behavior can be checked on the
host before flashing.

## What is written, and what is not

| Piece | State |
| --- | --- |
| Packet framing + CRC (`src/emg_packet.c`) | done, host-tested |
| 20–450 Hz band-pass + 50/150 Hz notches (`src/emg_filter.c`) | done, Q29 golden-vector checked against scipy; live on hardware |
| Feature extraction (`src/emg_features.c`) | done, 200 ms window / 50 ms hop and MAV/RMS/ZC/WL golden-vector checked for exact equality; live on hardware |
| ADC1 scan + DMA1 channel 1 acquisition | done and live-verified on IN0/IN1/IN4 |
| TIM3 2 kHz trigger | done; measured stream rate 2000.1 Hz |
| USB CDC INFO/RAW transmission | done and live-verified on `/dev/ttyACM0` |
| Acquisition main loop | RAW as before, plus filter → features → classifier → gate → INTENT per 50 ms hop. 57% RAM / 37% Flash; live-verified with zero packet loss and INTENT matching a host replay event for event |
| Wear-detect GPIO reads | done; one mask per half, shared by the RAW packet and by window validity so the two cannot disagree |
| Classifier inference | Q18 pure C scorer matches the Python reference on 297/297 hops of real recorded data; live on hardware |
| Event gate (`src/emg_gate.c`) | done; frozen counts, and a 1024-decision fixture the Python gate reproduces event for event |
| Activation threshold (`src/emg_activation.c`) | done and in the live loop; rest-relative, cross-checked step for step against the Python mirror. Frozen `K=3`, `shift=4` after an independent self-paced session, 6/6 correct; margin is 3 counts and should be re-measured |
| Host: probe/scope/record/replay/analyze/reference tools | done and live-used |
| Host: guided labelled capture GUI | implemented, headless-tested, and used for six complete balanced sessions across two donnings |
| Host: event-gate validation capture | two real complete sequences; the second passed independent validation 9/9 with pre-registered counts |
| Host: training pipeline | session-aware continuous-Q29 ridge-LDA LOSO plus Q18/C export implemented and measured |
| ROS 2 bridge | not started |

RAW remains available as the replayable source of truth, but it is no longer
the only live path. The MCU also runs filter, features, Q18 classification,
rest-relative activation, the frozen event gate, and INTENT every 50 ms. The
next work is ordinary-activity/soak measurement, a repository MCU/host event
comparison tool, the ROS bridge, proportional calibration, and final metrics.

### Guided labelled capture

Run the GUI as the normal desktop user, not with `sudo`. The active login must
already have access to `/dev/ttyACM0` through the `dialout` group.

```bash
cd /home/cold227/Documents/assistive_robot_ws
python3 firmware/tools/emg_guided_capture.py \
  --port /dev/ttyACM0 \
  --repetitions 5
```

The window previews the three raw ADC channels. `Start collection` begins one
continuous raw log after fresh three-channel INFO/RAW and stable electrode
contact are present. It then randomizes balanced trials using this fixed map:

| Label | Screen action |
| --- | --- |
| `REST` | relax completely / neutral wrist |
| `NEXT_TARGET` | wrist extension / wrist up |
| `CONFIRM` | make a fist |
| `ABORT` | wrist flexion / wrist down |

Only the three-second `HOLD` phase is labelled. `Pause` stops the prompt timer
and labels but keeps draining, displaying, and saving serial bytes; an action
interrupted by Pause is discarded and repeated after Resume. A short
unlabelled verification phase checks the labelled tail for delayed packet
loss, and any action with less than 90% of its expected frames is repeated.
After all valid trials the GUI stops automatically.

Each run creates `datasets/emg/session_<timestamp>/session.bin` plus
`session.json`. The binary is the full raw source of truth; JSON stores the
randomization seed, exact cumulative-frame label spans, pauses, rejected
attempts, contact/parser quality, and stream summary.

#### Event-gate validation capture

Use the same GUI with the independent event protocol:

```bash
python3 firmware/tools/emg_guided_capture.py \
  --protocol event-gate \
  --port /dev/ttyACM0
```

Its default schedule is `REST, gesture, REST, gesture, ... REST`, with three
randomized repetitions of `NEXT_TARGET`, `CONFIRM`, and `ABORT`. Each labelled
span lasts two seconds; the complete run takes about one minute. It saves to
`datasets/emg_event_gate/session_<timestamp>` so these timing-validation spans
cannot be mistaken for another classifier-training session. The software path
is tested. One real session completed 19/19 spans with 100% usable/contact
frames and zero parser loss, but independent replay passed 0/240 gates; best
clean was 3/9 and `NEXT_TARGET` was correct on 0/3 events. This first failure
is historical: onset hold-off was then diagnosed, and a separate session
passed the fixed gate 9/9. The later activation layer was independently checked
6/6.

### Host LDA baseline

`emg_train_lda.py` accepts only complete, balanced schema-v1 sessions. It
filters each channel once across the continuous recording with the same Q29
20–450 Hz + 50/150 Hz cascade used by the MCU reference, extracts
`ch0..ch2 × (MAV, RMS, WL, ZC)` on the global 400-sample/100-sample grid, and
keeps only windows wholly contained in an accepted ACTIVE span. Validation is
leave-one-session-out rather than a random window split, preventing overlapping
windows from the same recording leaking into both train and test data.

```bash
python3 firmware/tools/emg_train_lda.py datasets/emg \
  --output datasets/emg/lda_model.json \
  --fraction-bits 18 \
  --c-output firmware/src/emg_classifier_model.h
```

The 2026-08-14 five-session run loaded 100 trials / 5704 windows and excluded
one stopped session. Overall accuracy was 94.8% per window and 96.0% per trial.
`REST`, `NEXT_TARGET`, and `CONFIRM` were each 25/25 by trial; `ABORT` was
21/25, with four trials predicted as `CONFIRM`. Standardization is folded into
raw-feature Q18 affine coefficients. Q18 predictions match float on 5704/5704
source windows; the generated pure C scorer matches Python scores exactly on
its golden fixture and links in the ARM build. The JSON includes preprocessing,
class order, float/Q18 parameters, folds, and confusion matrices. `main.c` now
runs this classifier in the complete live path; host replay and MCU output
matched event for event on the same recording.

The classifier order is set in `TODO.md` and is deliberate: the Hudgins
feature set with an LDA/SVM baseline first, and a quantized MLP only if it
measurably beats that. The part can carry either — a 12-feature, 16-hidden,
4-class int8 MLP is 256 weights — so the constraint is not the silicon, it is
that a model with no baseline to beat cannot be judged.

## Tests

Neither suite needs the ARM toolchain or the board.

```bash
make -C firmware/test check  # packet/filter/features + cross-language fixtures
python3 -m pytest firmware/tools -q
```

Verified on 2026-08-14: all six C binaries (packet, filter, features,
classifier, event gate, and activation) passed, and the complete Python tools
suite reported **158 passed**.

Run them in that order. `make check` regenerates the packet, filter, feature,
classifier, event-gate, and activation fixtures. The Python suite independently decodes or recomputes
them and checks every field or score. The implementations share specifications
and model parameters, not runtime code, so disagreement exposes a boundary or
arithmetic error before flashing.

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
| `emg_analyze.py` | Reads a recording and judges electrode placement from the envelope correlation between channels. |
| `emg_guided_session.py` | Pure, headless timing/label/quality state machine for balanced guided collection. |
| `emg_guided_capture.py` | Tk GUI with three raw traces, Start/Pause/Resume/Stop, timed prompts, automatic retry/completion, continuous `.bin`, and frame-indexed `.json` labels. |
| `emg_train_lda.py` | Session-aware continuous-Q29 ridge-LDA training, leave-one-session-out evaluation, Q-format export, and generated C model header. |
| `emg_event_gate_replay.py` | Fold-specific Q18 full-timeline replay and candidate stable-window/REST-rearm/refractory sweep; unlabelled-gap events remain diagnostic only. |

`emg_record.py` stores **bytes, not decoded samples**. A recording of decoded
values is unrecoverable if the decoder had a bug; a byte log can be re-decoded
as often as needed, which is what `--replay` is for. It reports the sample
rate two ways — from the wall clock and from the firmware's own timestamps —
so a slow reader can be told apart from slow firmware.

```bash
python3 firmware/tools/emg_record.py --seconds 60 --out session.bin
python3 firmware/tools/emg_record.py --replay session.bin
python3 firmware/tools/emg_analyze.py session.bin
```

`emg_analyze.py` answers whether three channels carry three channels' worth of
information. It correlates the **envelopes**, not the samples: sEMG is a
stochastic interference pattern, so two electrodes over the same muscle still
show near-zero sample-to-sample correlation, and correlating samples would
score a redundant placement as fine. A test asserts exactly that trap.

Both need `/dev/ttyACM0` access, so the user must be in `dialout` and have
logged in again since being added.

```bash
python3 firmware/tools/emg_probe.py 10
python3 firmware/tools/emg_scope.py --channels 3 --rate 2000
```
