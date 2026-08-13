# Objective 3.5 — Hardware Bring-Up Log

How the Cheez sEMG board went from an unknown box to a firmware image that
builds, on 2026-08-13.

Kept separate from `firmware/README.md` on purpose. That file states what is
true about the hardware; this one records how each fact was established and
what was believed before. The wrong turns are the point — several of them were
the kind that produce plausible-looking data instead of an error.

There is no evaluation document for Objective 3.5 yet, and there should not
be: almost nothing has been measured. The firmware still has no acquisition
loop.

## The part number disagreed three ways

| Source | Says |
| --- | --- |
| Vendor schematic | STM32F103**C6**Tx — 32 KB Flash, 10 KB SRAM |
| Package marking | STM32F103**C8**T6 |
| `st-info --probe` | `flash: 65536`, `sram: 20480`, `chipid: 0x410`, `STM32F1xx_MD` |

`st-info` reads the factory-programmed size register on the die, so it and the
marking agree against the schematic. The working assumption is C8 and 64 KB,
recorded with a note to revisit if anything strange appears past 32 KB.

The habit worth keeping is asking for a second source at all. One reading felt
like enough until the schematic arrived and contradicted it.

## `A2` is `PA4`, and getting it wrong would not have failed

The board silkscreen was photographed and read as `A0..A5 = PA0..PA5`. The
schematic says otherwise: `PA2` and `PA3` are taken by the digital pins `D3`
and `D4`, so the analog run skips.

```
A0 -> PA0    A1 -> PA1    A2 -> PA4    A3 -> PA5    A4 -> PA6    A5 -> PA7
```

Four of six were wrong. The three-channel scan is IN0, IN1, **IN4**.

Configuring IN0/IN1/IN2 instead would have sampled `PA2` — a digital pin,
either floating or driven by whatever else uses it. It would not have thrown
an error. It would have produced a third channel of numbers that vary,
correlate with nothing, and look like a bad electrode.

This one came from the user obtaining the vendor schematic. Nothing in the
photograph could have settled it.

## ADC2 has no DMA — three grades of evidence for one claim

The claim was asserted first and supported afterwards, which is the wrong
order and produced a chain worth recording.

1. **A search engine's summary of forum posts.** Stated as fact in
   conversation. Not acceptable as support.
2. **An ST employee on ST's own forum**, citing RM0008 page 227 directly, and
   confirming that CubeMX showing no DMA tab for ADC2 is expected behaviour.
   Much stronger, still second-hand.
3. **The manual itself**, after the user asked "how do you know there is no
   DMA". RM0008 Rev 21 p. 227: *"Only ADC1 and ADC3 have this DMA capability.
   ADC2-converted data can be transferred in dual ADC mode using DMA thanks to
   master ADC1."* Table 78 independently confirms it — ADC1 sits on DMA1
   channel 1 and ADC2 appears nowhere in the mapping. And DMA2 exists only on
   high-density, XL-density and connectivity-line parts, so the medium-density
   C8 has neither ADC3 nor DMA2.

The user's challenge is what moved this from level 1 to level 3. The
conclusion did not change, but its status did, and only level 3 belongs in a
document.

Worth noting what the claim never affected: the design decision. Even if ADC2
had DMA, ADC1 scan mode is simpler than dual mode, and the ~6.8 us
inter-channel skew is irrelevant to amplitude features. A claim that cannot
change any decision should not have been stated with confidence in the first
place.

## The USB pull-up looked wrong and was fine

STM32F103 has no internal D+ pull-up, so the board must supply one. If it were
switched by a GPIO and we did not know which, custom firmware would never
enumerate — the highest-risk unknown going in.

The schematic shows `R13`, a fixed 10K from the 5V rail to `PA12`. No
transistor, so firmware does not have to drive anything.

10K to 5V looks wrong against the spec's 1.5K to 3.3V. Against the host's 15K
pull-down the two are equivalent: 3.3 × 15/16.5 = 3.0 V versus 5 × 15/25 =
3.0 V, both above the 2.7 V full-speed threshold.

The decisive evidence was not the arithmetic. **The board already enumerates**
as `0483:5740`. A device that works settles a question about whether it can
work.

The consequence is recorded in the README because it will cost an evening
otherwise: a fixed pull-up cannot be toggled to force re-enumeration, so
unplug and replug USB after flashing before concluding new firmware is broken.

## The factory firmware settled a question that was going to be argued

The plan already said to write custom firmware. Two measurements turned that
from a preference into a requirement.

**499.9 Hz.** 5001 sample lines in 10.00 s, 1788 B/s of ASCII. Nyquist is
therefore 250 Hz against a planned 20–450 Hz band, so content above 250 Hz
would alias into the features.

**The output is not raw.** Values are signed and centred on zero, and 500
consecutive samples read exactly 0 at rest. So it applies DC removal and some
form of squelch. Not knowing what was removed is worse than the removal:
nothing downstream can be characterized or reproduced.

Separately, the analog chain passed: at rest the capture sits within ±2
counts, gripping swings it to ±1141. A dynamic range over 500× means
electrodes, module, wiring, and ADC input all work. That was the item actually
blocking everything else, and it is the one thing that was verified before any
firmware discussion mattered.

## Two CubeMX behaviours that are backwards from intuition

**Scan Conversion Mode cannot be enabled directly.** It is derived: set
`Number Of Conversion` above 1 and Scan flips to `Enabled` on its own and
stays greyed. Half an hour was lost trying to set the dependent value first.

**Timer registers are N−1.** `PSC = 35` divides by 36 and `ARR = 999` counts
1000, because the counters start at zero. Filling in 36 and 1000 would give
72e6/37/1001 = 1943.9 Hz — 2.8% off, with no error, and with every window
duration and filter cutoff quietly shifted to match.

## The Flash estimate was wrong by a factor of two

"HAL + USB CDC middleware is roughly 30–40 KB of the 64" was stated as a
constraint. The first build says **18 700 B at `-Os`**, 28.5%. The 39 304 B
figure that the estimate matched is the `-O0 -g3` debug build.

RAM is the tighter resource and was not the one flagged: 7 704 B before any
per-channel state exists, projecting to about 11.2 KB once three channels of
filter and feature state, the DMA double buffer, and a transmit buffer are
added.

An estimate that happens to match the debug build is not validated by that.

## The errors that would not have announced themselves

Collected because this is the whole theme. Every one of these compiles, runs,
and produces data.

| Mistake | What you would see |
| --- | --- |
| ADC channels IN0/IN1/**IN2** | Third channel varies plausibly, correlates with nothing |
| ADC prescaler `/4` at 72 MHz | 18 MHz against a 14 MHz ceiling; conversions quietly wrong |
| `SYS -> No Debug` | Firmware runs; ST-Link can never attach again |
| `PSC = 36`, `ARR = 1000` | 1943.9 Hz called 2000 Hz; every frequency claim shifts |
| Sampling time too short | Channel-to-channel crosstalk that reads as correlated muscles |
| Wear-detect pins floating | Unplugged column reads as "worn"; noise trains as gesture |
| Trusting the silkscreen | See above |

The pull-down on `PA8`/`PA2`/`PA3` and the fail-closed reading of wear state
are there for the sixth row specifically.

## Lessons

- **Ask what the source is before believing the claim.** A forum summary, a
  cited page number, and the page itself are three different things, and only
  the third is support.
- **State a claim only when it can change a decision.** The ADC2 question was
  argued at length and never affected the design.
- **A photograph of a silkscreen is not a schematic.** Four of six pins were
  wrong, and none of them would have failed loudly.
- **A device that works answers questions about whether it can work.** The USB
  pull-up arithmetic was reassuring; the enumeration was decisive.
- **An estimate matching one build configuration is not a measurement.**
- **Prefer errors that fail loudly.** Every deliberate choice above — the
  pull-downs, the timer trigger instead of continuous conversion, refusing
  partial feature windows — is buying a loud failure instead of a quiet one.
