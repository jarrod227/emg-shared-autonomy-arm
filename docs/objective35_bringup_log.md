# Objective 3.5 — Hardware Bring-Up Log

How the Cheez sEMG board went from an unknown box to three channels of clean
muscle signal, on 2026-08-13 and 2026-08-14.

Kept separate from `firmware/README.md` on purpose. That file states what is
true about the hardware; this one records how each fact was established and
what was believed before. The wrong turns are the point — several of them were
the kind that produce plausible-looking data instead of an error.

There is no evaluation document for Objective 3.5 yet, and there should not
be. What is measured here is that the acquisition works and the electrodes
are placed usefully. Nothing is yet measured about what the objective is for:
no classifier exists, so there is no accuracy, no latency, and no false-
trigger rate to report.

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

## Getting a usable signal out of it

Bring-up ended with a firmware image that built. Getting three channels of
real muscle activity out of it took a further six recordings, and almost all
of the difficulty was in the measuring rather than in the hardware.

### The band-pass could never remove the mains, and I argued against fixing it

The first three sessions all produced confident placement verdicts. A
spectrum of one resting segment says 98.7%, 99.7% and 99.9% of in-band power
sat at 50 Hz, peaking at exactly 50.0 Hz. Rest was not muscle at rest; it was
the room. 50 Hz is squarely inside 20-450 Hz, so the filter passed it
untouched and it presented as a large steady amplitude.

Asked whether a notch would fix it, I said no: at 97% hum there would be
nothing left underneath. That was wrong, and wrong for a reason worth
naming: power and amplitude are not the same fraction. 3% of the **power**
left is 17% of the **amplitude**, since sqrt(0.03) = 0.17, and that 17% was
a real signal all along. Notching the same recording took channel 1 from
a 1.5x rest-to-contraction contrast to 6.4x, and rest fell from a steady MAV
of 200 to single digits.

I had also framed reducing hum at the source as a precondition. It is not;
it is an independent improvement. The notch made the existing data usable
immediately.

### Which harmonics, measured rather than assumed

| Notches | Extra sections | ch1 contrast | ch2 contrast |
| --- | ---: | ---: | ---: |
| none | 0 | 1.5x | 1.5x |
| 50 | 1 | 5.4x | 9.9x |
| 50, 100 | 2 | 5.4x | 9.8x |
| **50, 150** | **2** | **6.3x** | **22.4x** |
| 50, 100, 150 | 3 | 6.4x | 22.5x |
| 50 to 250 | 5 | 6.4x | 23.3x |

The fundamental captures most of it, the third harmonic gives a second clear
jump, and the second harmonic contributes almost nothing. Neither "just notch
50" nor "notch all the harmonics" would have been the right answer, and the
table is why the shipped cascade is 50 and 150 in two sections.

Narrow notches are the most quantization-sensitive shape in the design, so
the fixed-point version was validated before any C was written: every pole
inside the unit circle at a maximum radius of 0.9974, 50 and 150 Hz at gain
0.000, and 80/200/300 Hz passing at 0.999/0.994/0.949.

### Separating a bad electrode from a bad position

One channel stayed weak through several placements while correlating 0.92
with its neighbour -- the signature of an electrode coupling badly enough
that it only picks up the loudest nearby muscle by volume conduction.

Swapping the two bands' positions on the arm, with each band still on its own
board column, distinguishes the two explanations in one recording: if the
weakness follows the band it is hardware, and if it stays at the spot it is
the position. It stayed at the spot. Both bands were fine, and that spot on
the forearm had little muscle under it -- most likely too far toward the
wrist, over tendon.

### Four verdicts that measured something other than what they claimed

The recurring failure of the session, and the reason the analyser now carries
three quality gates it did not start with.

| Reported | Actually | Root cause |
| --- | --- | --- |
| 1808 Hz, then 1940 Hz | 2000.1 Hz | Packets arriving before the first INFO were dropped from the frame count while the time they occupied stayed in the denominator |
| "all pairs distinct" | one channel was dead | A silent channel correlates with nothing, which is indistinguishable from being usefully independent |
| "all pairs distinct" | 99% of the signal was mains | Amplitude alone cannot tell muscle tone from hum |
| "redundant, move the bands" | placement was good | Rest periods make every channel go quiet together, and that shared on/off swing dominated the coefficient: one pair read 0.71 instead of 0.04 |

None of these was a threshold set wrong. In every case the number was
arithmetically correct and answering a different question than the one asked.
The fixes were, in order: replay the held packets instead of discarding them;
refuse a verdict when a channel is silent; refuse it when the channel is
mains-dominated after filtering; and correlate only the windows where the
muscle is working.

The recorder storing bytes rather than decoded samples paid for itself here.
Two of those four were fixed by re-decoding an existing recording, without
repeating the experiment.

### Where placement ended up

| | |
| --- | --- |
| Correlation on active windows | ch0-ch1 0.63, ch0-ch2 0.04, ch1-ch2 0.66 |
| Mains before filtering | 8%, 8%, 7% -- from 97% at worst |
| Clipping | none |
| Electrode contact | 100% of frames on all three |
| Fist | ch0 195, ch1 46, ch2 95 -- normalized 0.55 / 0.14 / 0.32 |
| Open | ch0 77, ch1 37, ch2 122 -- normalized 0.33 / 0.15 / 0.52 |
| Pattern distance between the two | 0.30 |

Channel 1 stays the weakest of the three and was left that way deliberately:
two well-separated channels plus a mediocre one is enough to start training,
and whether the third is worth more effort is a question measured
classification accuracy can answer and further fiddling cannot.

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
