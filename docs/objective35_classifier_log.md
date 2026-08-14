# Objective 3.5 — Classifier and Event-Gate Log

How a 96% classifier produced three correct events out of nine, on 2026-08-14.

Companion to `docs/objective35_bringup_log.md`, which covers the hardware and
stops where a clean three-channel signal exists. This file starts there and
covers labelling, the LDA, the fixed-point port, and the discrete-event gate.

Nothing here is frozen. No gate configuration is approved, no accuracy claim
survives contact with an independent session, and the one event-gate recording
that exists has since been used for tuning, so it is no longer an unbiased
test set. What is settled is why the first honest measurement failed.

## The 96% measured within-donning repeatability

Five balanced classifier sessions, 100 accepted trials, 5704 feature windows.
Leave-one-session-out ridge LDA measured 94.8% window and 96.0% trial
accuracy: `REST`, `NEXT_TARGET` and `CONFIRM` 25/25 by trial, `ABORT` 21/25
with four trials reading `CONFIRM`.

Their timestamps are the problem.

| Session | Started |
| --- | --- |
| `session_20260814_050815` | 05:08:15 |
| `session_20260814_052235` | 05:22:35 |
| `session_20260814_053654` | 05:36:54 |
| `session_20260814_055033` | 05:50:33 |
| `session_20260814_055430` | 05:54:30 |

Forty-six minutes end to end, with the electrodes never removed. Holding one
of those out and training on the other four does not hold out a session in any
sense that matters; it holds out a segment of one recording. The number is
arithmetically correct and answers a question nobody asked — the same failure
that produced four wrong verdicts during bring-up, now the fifth.

The first genuinely independent recording, `session_20260814_110427`, was
taken about five hours later after the electrodes were reapplied.

## What the independent session was not

Acquisition quality removed most of the candidate explanations before any
analysis started.

| Checked | Measured |
| --- | --- |
| Packets | 0 lost, 0 malformed, 0 duplicated, 0 time-reversed |
| Frames | 148 352 at 2000.2 Hz |
| Electrode contact | 100% on all three channels |
| Usable fraction | 1.0 |
| Mains power after filtering | 8% / 10% / 8% |
| Q18 against float | 100% agreement on every validation window |
| Rejected attempts | 0 |

So it was not the link, not the electrodes, not the quantization. It swept
0/240 gate configurations, best clean 3/9, with `NEXT_TARGET` correct on 0/3.

## Two explanations that were tested and are wrong

Both were plausible, both had a mechanism, and both are refuted by the data
rather than by argument. They are recorded because the cost of each was one
script, and the cost of believing either would have been a redesign.

### Amplitude gain drift

Every class came out of the independent session two to eight times louder than
in training, `REST` included — total MAV across channels sat at 28–35 against
15–16. That is a real difference: a different baseline, flat across the whole
session, not a drift within it and not an artefact of rest spans starting
before the previous contraction had decayed.

LDA is not scale invariant, so this looked decisive. It is not the cause.

| Feature transform | Labelled-window accuracy |
| --- | --- |
| Raw features | 0.819 |
| Per-window, amplitudes over that window's total MAV | 0.791 |
| Per-session, amplitudes over the session's 95th-percentile total MAV | 0.803 |

Both normalizations made it slightly worse. The elevated baseline is real,
worth noting, and did not drive the failures.

### Effort recruiting the antagonist

The independent session's gestures were performed harder, and harder
contractions do recruit more co-contraction, which would pull an extension
gesture toward the flexor channel. Testing it inside the training set, where
effort varies naturally:

| `NEXT_TARGET` training trials | Total MAV | ch0 share |
| --- | --- | --- |
| Weakest eight | 97.7 | 0.094 |
| Strongest eight | 180.0 | 0.096 |

Pearson r between total MAV and ch0 share is +0.055 over 25 trials spanning
70 to 220. Effort does not move the channel balance at all. The trial that
actually fails sits at 173, in the middle of that range, with a ch0 share of
0.406.

## What the errors were: windows that straddle rest and contraction

Aligning every misclassified window with its position inside the labelled
span:

| | First 1.0 s | After 1.0 s | Early share |
| --- | --- | --- | --- |
| All active trials | 119 | 34 | 0.778 |
| Excluding trial 13 | 103 | 16 | 0.866 |

The classifier sessions used `prepare = 2.0 s`, so a labelled span began two
seconds after the prompt and contained steady state only. The event-gate
session used `prepare = 0.5 s`, so its spans contain the onset — and a gate in
real use has no choice but to run through every onset. A 200 ms feature window
that straddles rest and contraction is a mixture the classifier has never
seen, and it lands on `CONFIRM`, the highest-energy class. `CONFIRM` itself
was never confused, because for `CONFIRM` the onset guess happens to be right.

The event-gate timing is the realistic one. The training protocol is the one
that was wrong, and it was wrong in a way that inflated its own score.

## The fix, and what it costs

`EventGate` now starts an onset hold-off when the stream leaves `REST`.
Hold-off windows are discarded rather than scored: no candidate run
accumulates, `ABORT` is covered too, and a `REST` window cancels the hold-off
so a twitch cannot leave it half spent. It is deliberately not a confidence
threshold — the onset predictions are confident, not marginal, so a margin
test would pass them.

Best configuration at each hold-off, over 1200 swept configurations:

| Hold-off | Clean | Wrong | Missed | Duplicate | REST false | Off-trial | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3/9 | 6 | 3 | 3 | 1 | 0 | 1036 ms |
| 4 | 7/9 | 2 | 0 | 2 | 0 | 1 | 1377 ms |
| 8 | 7/9 | 2 | 0 | 2 | 0 | 1 | 1287 ms |
| 12 | 8/9 | 1 | 1 | 0 | 0 | 0 | 1598 ms |
| 16 | 8/9 | 1 | 1 | 0 | 0 | 0 | 1648 ms |

Three clean trials to eight, from one change, with nothing else retuned. The
cost is one hold-off period of latency on every event: 0.6 s at hold-off 12,
p50 1.06 s and p95 1.60 s measured from the start of the labelled span.

The sweep grid is coarse on purpose. Nine active trials cannot separate 0.60 s
from 0.80 s, and a finer grid would only offer more ways to fit this session's
noise. Zero of the 1200 configurations meet the acceptance criteria, which
require 9/9 with no wrong, missed, duplicate, or off-trial events.

### The prototype was optimistic because it shared a code path

The first hold-off experiment did not add a state; it pushed a synthetic
`REST` into the gate during the hold-off. That incremented the rest run, so
the gate quietly re-armed itself mid-hold-off, and the prototype reported 8/9
with zero wrong and zero missed. The same hold-off length under the real
implementation gives 8/9 with one wrong and one missed.

A shortcut that reuses the real component's entry point will report the answer
the shortcut implies, not the answer the design implies. The number moved in
the flattering direction, which is the direction that does not get questioned.

## The remaining failure looked like donning drift, and was not

The remaining failure is a single trial, and it is not a badly performed one.
Channel balance as a share of total MAV, over settled windows:

| Class | Training mean | Independent session, three trials |
| --- | --- | --- |
| `CONFIRM` | 0.63 / 0.14 / 0.23 | 0.58/0.14/0.27, 0.61/0.14/0.25, 0.61/0.15/0.24 |
| `NEXT_TARGET` | 0.09 / 0.11 / 0.80 | 0.37/0.07/0.56, 0.29/0.06/0.65, 0.41/0.07/0.52 |

`CONFIRM` reproduces almost exactly, which rules out the electrodes having
moved: a shifted band would move both classes. `NEXT_TARGET` moved its ch0
share by a factor of three to four, in the same direction on all three trials
— a session-level shift, not trial noise, and toward `CONFIRM`, which is
ch0-dominant. Trial 13 has the largest ch0 share of the three and is the one
that crosses. Trials 1 and 7 pass narrowly and should not be read as safe.

What cannot be determined from this data is why. A wrist-up performed with the
fingers partly flexed, or with the forearm supported differently, would both
produce this and are not separable from three trials of one session. Effort is
excluded, as above.

The conclusion drawn at that point was that donning variability is the open
accuracy risk, and that training would have to span several separate donnings
before any cross-session number meant anything. That conclusion was wrong, and
it was wrong for a reason worth keeping: it generalized from one independent
donning. A session-level shift measured on one session is still a sample of
one at the level that matters.

### A third donning settled it

A classifier session was recorded at 12:26, after the electrodes had been fully
removed and reapplied, with the physical gesture pinned first: forearm flat and
supported the same way for all three actions, fingers relaxed and straight
during `WRIST UP`, no wrist involvement during the fist, moderate effort.

| `NEXT_TARGET` ch0 share | Total MAV |
| --- | --- |
| Donning A, 05:08-05:54 — 0.09 | 138.9 |
| Donning B, 11:04 — 0.36 | 253.2 |
| Donning C, 12:26 — 0.12 | 196.8 |

| Trained on | Tested on | Overall | REST | NEXT | CONFIRM | ABORT |
| --- | --- | --- | --- | --- | --- | --- |
| A | C | 0.985 | 1.00 | 1.00 | 0.97 | 0.97 |
| A | B | 0.819 | 0.97 | 0.48 | 0.88 | 0.58 |
| A + C | B | 0.845 | 0.98 | 0.55 | 0.88 | 0.65 |

Cross-donning generalization is not the problem. A model fit on donning A,
seven hours and one full redonning earlier, classifies donning C at 98.5%
overall and 100% on the class that supposedly did not survive redonning.

Donning B is the outlier, not the rule. Adding donning C to the training set
barely moves B (`NEXT_TARGET` 0.48 to 0.55), which is what an outlier does and
not what a point on a donning-variability manifold does: more coverage of the
manifold would have helped substantially. The likeliest remaining explanation
is the one the earlier section could not exclude — that session's `WRIST UP`
was executed differently — and pinning the gesture definition preceded the
donning that behaved.

This is not proof that pinning the definition caused the recovery. Donning C
differs from B in more than one way, and one recovered session is as much a
sample of one as the failure was. What it does establish is that redonning
alone does not break the classifier, so multi-donning collection is not the
blocking task it was written up as.

## The independent validation, and what it is worth

An event-gate session was recorded on donning C, minutes after its classifier
session. The gate configuration was not chosen on it: `stable_windows = 5`,
`rest_rearm_windows = 4`, `refractory_windows = 5`, `abort_stable_windows = 1`,
`onset_holdoff_windows = 12` were fixed on the donning B sweep and applied
unchanged. The model was fitted on donning A alone, so donning C entered
neither the fit nor the parameter selection.

| | |
| --- | --- |
| Active trials | 9 |
| Clean correct trials | 9 |
| Missed / wrong / duplicate | 0 / 0 / 0 |
| REST spans | 10 |
| REST false events | 0 |
| Off-trial events | 0 |
| Invalid windows | 0 |
| Per class | `NEXT_TARGET` 3/3, `CONFIRM` 3/3, `ABORT` 3/3 |
| Latency p50 / p95 | 1064 ms / 1167 ms |
| Q18 against float | 1.0000 |

The pass is not a knife edge. 220 of the 1200 swept configurations pass on this
session, and every one of the twelve neighbours of the frozen configuration —
all four refractory values crossed with all three `abort_stable_windows` values
— passes 9/9. Hold-off is the exception, and it confirms itself on a session
that had nothing to do with its selection: hold-off 0, 4, and 8 pass 0/240
each, and only 12 and 16 pass at all.

What this does not establish: 9 active trials and roughly 20 seconds of
labelled REST, in one 72-second scripted recording, cannot support a
false-triggers-per-minute number. That REST is prompted relaxation, not
ordinary activity, and the recording contains no incidental movement at all.
Everything above is offline replay on a PC; the firmware still emits RAW only.
This is sequence integrity, which is a necessary condition and not deployment
approval.

## `ABORT` is separable only after about a second

`ABORT` scored 0.51 on labelled windows in that session while every other class
was above 0.9, and all three `ABORT` trials still produced one correct event.
Both facts have the same cause, and it is not a weak class.

| Same donning, minutes apart | `ABORT` window accuracy |
| --- | --- |
| Classifier protocol, labelled span is steady state | 0.97 |
| Event-gate protocol, labelled span contains the onset | 0.51 |

`WRIST DOWN` and `MAKE A FIST` are both anterior-compartment gestures — wrist
flexion and finger flexion, from adjacent and partly overlapping muscle groups.
Both load ch0 heavily. `WRIST UP` is extension, a different compartment
entirely, which is why it is the easy class.

| Class | ch0 | ch1 | ch2 |
| --- | --- | --- | --- |
| `NEXT_TARGET` | 0.09 | 0.11 | 0.80 |
| `CONFIRM` | 0.63 | 0.14 | 0.23 |
| `ABORT` | 0.49 | 0.37 | 0.14 |

Ablation agrees: zeroing ch2 takes `NEXT_TARGET` from 1.00 to 0.41, and zeroing
ch1 costs `ABORT` more than any other class. What separates `ABORT` from
`CONFIRM` is essentially the ch1/ch0 ratio, 0.76 against 0.22.

That ratio needs more than a second to establish.

| Time since span start | `ABORT` ch1/ch0 | `CONFIRM` ch1/ch0 |
| --- | --- | --- |
| 0.25 s | 0.24 | 0.25 |
| 0.50 s | 0.26 | 0.17 |
| 0.75 s | 0.32 | 0.13 |
| 1.00 s | 0.43 | 0.10 |
| 1.25 s | 0.44 | 0.10 |

At 0.25 s the two gestures are the same measurement. The opening burst of a
wrist flexion is ch0-dominated; ch1 catches up only during the sustained hold.
The classifier was trained on windows from two seconds after the prompt, where
the ratio has long settled, so it calls the ambiguous phase `CONFIRM` with
confidence rather than with a small margin.

This is also why hold-off 16 outperformed 12 by a wide margin — 160/240 against
60/240 — for a reason independent of the score: 0.8 s covers more of the
ambiguous phase than 0.6 s does. The frozen value stays 12 because that is the
one that was validated. Moving to 16 is a change, and a change has to be
validated on a session that did not motivate it.

### The margin is one window

Trial 11 of the validation session, after hold-off ends at 0.74 s:

```text
CONFIRM[0.19-0.94] ABORT[0.99] CONFIRM[1.04] ABORT[1.09-1.29] ...
└─ hold-off covers 0.19-0.74 ─┘
                    0.79 0.84 0.89 0.94  =  four consecutive CONFIRM
```

`stable_windows` is 5. One more 50 ms window of `CONFIRM` and that trial emits
`CONFIRM` instead of `ABORT` — the safety gesture read as a confirmation, which
is the worst confusion this system can make. It passed with a margin of one
window, on a sample of three `ABORT` trials, and that margin has never been
measured.

`ABORT` is therefore recorded as a known weakness with a quantified reason, not
as a class that works because the gate rescued it. Channel 1 was left as the
weakest of the three during bring-up as a deliberate deferral; this is the
first measurement that argues for revisiting its position.

## Putting it on the MCU

The gate existed only in Python, so it was ported to `src/emg_gate.c` branch
for branch and checked against a 1024-decision fixture the Python gate
reproduces event for event. Then the firmware loop was wired up: each consumed
half runs filter → features → Q18 classifier → gate → one INTENT packet per
50 ms hop, with RAW still streaming so the host can replay the identical
samples.

Two 2026-08-14 recordings, both after the board had been running for minutes.

| | 15 s, electrodes off | 60 s, worn, six gestures |
| --- | --- | --- |
| Frames | 30 016 at 1999.3 Hz | 120 000 at 1999.5 Hz |
| Lost / malformed / duplicated | 0 / 0 / 0 | 0 / 0 / 0 |
| Electrode contact | ch1 detached 13% of frames | 100% |
| INTENT packets | 301 = 20.05 Hz | 1200 = 20.0 Hz |
| INTENT timestamp step | exactly 50 000 µs, every step | same |

The first result that mattered was the negative one: adding the whole DSP to
the loop cost no packets. The timing budget was the obvious way this could
have failed and it did not.

### The classifier calls floating electrodes a gesture 6% of the time

With the band off, 14 of 241 settled windows classified as `NEXT_TARGET` and
one as `CONFIRM`. The gate emitted nothing: scattered decisions never reach a
run of five, and the wear mask invalidated 43 windows outright. Worth writing
down as a number rather than an impression — it is the only false-trigger
figure that exists so far, and it says the classifier alone would not be safe
to act on.

## Verifying the MCU against the host, and two ways it misleads

### Confidence disagrees; events do not

`confidence` is a deterministic function of the scores, so it works as a probe
even when nothing happens. On the idle recording the MCU and a host replay
agreed on 263 of 297 hops. Chasing that gap produced the useful part.

The frozen C coefficient table and the Python design are byte-identical, and no
filtered sample came near int16, so neither was the cause. Running the same
recorded frames through the firmware's own C sources on the host settled it:

| Comparison | Agreement |
| --- | --- |
| host C vs Python reference | **297 / 297** |
| host C vs MCU | 263 / 297 |

The implementations agree exactly on real data. What differs is state: 73% of
the MCU mismatches fall in the first 3000 frames, where the host replay's
filters are still settling from zero while the MCU's have run for minutes.
After frame 6000 the residual is about 4% of hops, differing by at most 5
confidence units, in both directions.

That residual is not a bug to fix. The notch sections sit at Q = 30, pole
radius ≈ 0.9974, which is exactly where fixed-point IIR limit cycles live: two
instances with different histories can differ by an LSB indefinitely rather
than reconverging. **A recording that starts mid-stream therefore cannot be a
bit-exact check of the MCU**, and asking for one would be asking the arithmetic
for something it does not offer.

Events are immune to it, and that is the point. The gate needs five consecutive
agreeing decisions, which an occasional LSB divergence cannot manufacture or
destroy. Verify events, not scores.

### The replay grid must be aligned to the device

On the 60 s worn recording the MCU emitted six events and the host replay
emitted six events, same commands, same order — and every one of them
disagreed, by +52 or −48 frames.

The recording began at absolute frame 936 352. The firmware's hops land on
multiples of 100 counted from reset; a replay that starts counting from the
first recorded frame lands on multiples of 100 offset by `936352 mod 100 = 52`.
The two sides were computing different 400-sample windows. Aligning the replay
grid to absolute frame numbers:

| Frame | t | MCU | Host |
| --- | --- | --- | --- |
| 952 800 | 476.40 s | `NEXT_TARGET` | `NEXT_TARGET` |
| 967 500 | 483.75 s | `CONFIRM` | `CONFIRM` |
| 983 400 | 491.70 s | `ABORT` | `ABORT` |
| 999 400 | 499.70 s | `NEXT_TARGET` | `NEXT_TARGET` |
| 1 015 700 | 507.85 s | `NEXT_TARGET` | `NEXT_TARGET` |
| 1 037 700 | 518.85 s | `ABORT` | `ABORT` |

Event for event, exact. The misalignment is worth recording because of how it
presented: not as an obvious offset but as six clean events on each side that
happened to share no timestamps, which reads like a disagreement and is not
one.

## The first real-use failure: a missing activation threshold

The fifth gesture was a fist and the firmware emitted `NEXT_TARGET`. The
classifier was not wrong about it — all 43 windows of that contraction
classified as `CONFIRM`, channel balance 0.77/0.08/0.16. The event fired at
507.85 s and the fist began at 507.95 s.

| t | Prediction | ch0 | ch1 | ch2 | Total MAV |
| --- | --- | --- | --- | --- | --- |
| 507.00 | `REST` | 4 | 5 | 40 | 49 |
| 507.05 | `NEXT_TARGET` | 4 | 5 | 41 | 50 |
| 507.50 | `NEXT_TARGET` | 4 | 8 | 63 | 75 |
| 507.85 | `NEXT_TARGET` | 19 | 7 | 53 | 79 ← event |
| 508.20 | `CONFIRM` | 564 | 57 | 115 | 736 |

Before making the fist the wrist extended slightly — a preparatory movement,
about 0.85 s of low-amplitude extensor activity. The classifier read it
correctly as extension. Seventeen windows outlasted the twelve-window hold-off,
five more accumulated, and the gate fired 0.15 s before the intended gesture.

Every rule executed as designed. The design is missing one.

| | Total MAV |
| --- | --- |
| Rest | ≈ 30 |
| Preparatory movement | ≈ 65 |
| Intended gestures | 318 – 736 |

A factor of ten separates intent from incidental movement, and the gate does
not look at amplitude at all — only at which class the shape resembles. A
hold-duration rule would also have caught this one, but with a margin of 1.2×
against amplitude's 10×.

Three things follow, and they are separate.

The MCU and the host agreed on this misfire too, so the pipeline is faithful
and the defect is in the design. Those are different findings and collapsing
them would lose one.

The 9/9 event-gate result is not overturned, but its scope is narrower than it
read. In that protocol the user acts on a cue and goes straight into the
gesture. **Self-paced use produces failure modes that cued protocols cannot
generate**, and this one appeared in the first sixty seconds of it.

The threshold has to be measured and has to be relative — some multiple of a
recent rest baseline, not an absolute count. Donning B ran at twice the
amplitude of donning A, so an absolute floor tuned on one donning would be
either useless or crippling on another.

It is also worth stating what did not go wrong. A preparatory movement before a
gesture is ordinary motor behaviour, and this system is meant for users with
less motor control rather than more. Pinning the physical definition of a
gesture is defining the vocabulary and is legitimate; asking the user not to
move before moving is asking them to compensate for a missing threshold, and
would have closed a real defect as operator error.

## Fixing it: an activation threshold, and two ways of choosing it wrong

The stage rewrites low-activation non-`REST` decisions to `REST` before the
gate sees them — rewriting rather than dropping, because `REST` is what lets
the gate re-arm, and because "shape without activation" is what rest means
here. The threshold is a multiple of a baseline tracked by an integer EMA over
windows the classifier itself called `REST`, so a long sub-threshold movement
cannot drag its own threshold up behind it. It is relative and not absolute:
donning B rested at twice donning A, so a floor tuned on one is either useless
or crippling on the other.

Replayed on the recording that produced the defect, the fifth event changes
from `NEXT_TARGET` to `CONFIRM` and moves from 507.85 s to 508.85 s — after
the fist actually began at 507.95 s, so the preparation was suppressed rather
than the outcome coincidentally flipped. The other five events keep their
commands and their timestamps to within one hop.

### The sweep read the wrong session first

The first joint sweep reported that every factor which fixed the defect also
dropped the frozen 9/9 acceptance to 8/9 with one wrong event — 0 of 60
configurations passing both. It was measuring donning **B**:
`load_event_gate_timelines` returns every complete session in the directory
and the harness took `folds[0]`, which is the outlier session that passed
0/240 gates and was never expected to reach 9/9. Naming the validation session
by path instead:

| | K = 2 | K = 3 | K = 4 | K = 5 | K = 6 |
| --- | --- | --- | --- | --- | --- |
| Defect recording, 6 events correct | yes | yes | yes | yes | **no** |
| donning C, 9/9 with no wrong/missed | yes | yes | yes | yes | **no** |

19 of 60 configurations pass both, spanning K = 2..5 across every swept shift.
Above that the threshold starts suppressing real gestures and events go
missing. (Donning B, incidentally, goes from a best of 3/9 to 8/9 under the
same stage — worth revisiting when that session's anomaly is explained.)

### Passing without margin, again

The events do not separate K = 2..5, so the mechanism has to. Two selectors
were tried and one was discarded.

A per-window margin is meaningless here: real gestures legitimately have
onset and offset windows below threshold, and spurious windows may sit above
it, because the gate integrates over five decisions rather than judging one.

A run-length margin — longest run of consecutive above-threshold windows — is
better posed but unstable in practice. At fixed K it jumped 6, 7, 29, 30
across shifts, because a single marginal window splits one run into two. It
measures proximity to the threshold, not robustness, and was dropped.

What settled it was looking at the defect directly.

| | K = 2, threshold 70 | K = 3, threshold 105 |
| --- | --- | --- |
| Preparatory movement, 50–79 counts | four windows cross | none cross |
| Why no event fires | the four fragment into runs of two, short of five | nothing to fragment |

K = 2 passes by fragmentation. That is the same shape as the `ABORT` trial
that cleared its threshold by one 50 ms window: a pass with no margin behind
it, which reads identically to a pass with one. K = 3 clears the whole
movement outright and still sits two steps below where real gestures break.

Neither number was frozen at that point. Both came from two recordings — one
used to find the defect, one already spent as an acceptance set — and choosing
among 19 passing configurations on those same two is the shape of overfitting
this project has been careful about elsewhere.

### An independent session froze it

A third recording, self-paced, six gestures, took no part in choosing K. All
six events fired correctly with no spurious or missing event, which is the
acceptance test. It also went further than a pass/fail count: the activation
stage suppressed 88 windows across 17 low-amplitude episodes, three of them
21, 17, and 11 windows long — the first two longer than twelve hold-off
windows plus a five-window run, so each would have fired a spurious event on
the previous firmware, and both were `ABORT`-dominated.

The tightest of those episodes peaked at 90 total MAV against a threshold of
93, over 21 windows. `K = 2` — which the joint sweep had passed only by
fragmenting a shorter episode into runs of two — sets that threshold at 62 and
would have let this one through whole, firing a false `ABORT`. The independent
session did what the sweep alone could not: it ruled out the runner-up rather
than merely failing to distinguish it.

`EMG_ACTIVATION_FACTOR = 3` and `EMG_ACTIVATION_BASELINE_SHIFT = 4` are frozen
on this basis. The margin is thin — three counts separated the tightest
episode from its threshold — and is recorded as such rather than rounded up to
"validated." A fourth recording that measures that margin again, ideally with
a session that pushes closer to it on purpose, would say whether three counts
is a robust separation or a second lucky number.

## Lessons

- **Held-out is a property of the split, not of the file name.** Five
  recordings from one donning are one session with four folds of self-flattery.
- **Validate the protocol as well as the model.** Excluding onsets from the
  labelled span removed the only windows the gate would find difficult, and the
  accuracy figure never mentioned it.
- **A shortcut that shares the real code path will lie in your favour.**
- **Refute your own explanation before building on it.** Gain drift and
  co-contraction each cost one script to kill and would have cost a redesign
  to believe.
- **Say which alternatives the data cannot separate.** Three trials of one
  session cannot distinguish a gesture-execution change from a posture change,
  and writing "unknown" is cheaper than a wrong attribution.
- **One independent session is one sample.** Naming a cause "donning
  variability" turned a single anomalous recording into a property of the
  method, and the next donning refuted it in five minutes of collection. The
  effect was real and reproducible within that session; the explanation was
  not tested at all.
- **Test the cheap explanation before planning around the expensive one.** The
  refutation cost one redonning and a four-minute session. The plan it replaced
  was several days of collection.
- **A metric falling can be the measurement improving.** `ABORT` went from 0.97
  to 0.51 because the protocol stopped excluding the hard part. The 0.97 was
  the useless number: it sat there unchanged while the system delivered three
  correct events out of nine.
- **Passing is not the same as having margin.** Nine of nine, and one 50 ms
  window from the worst available failure. Report both or the report is wrong.
- **Ask the arithmetic for what it can give.** Bit-exact agreement between two
  fixed-point IIR filters with different histories is not available at any
  effort. Events survive an LSB; scores do not. Pick the quantity that the
  implementation can actually be held to.
- **A correct classification can still be a wrong command.** The system read
  the muscle right and acted on a movement that was not an instruction. Nothing
  in an accuracy figure covers that distinction.
- **Cued protocols cannot produce self-paced failures.** Sixty seconds of
  unscripted use found a defect that a passing scripted validation could not
  have, because being told when to move removes the preparation.
- **"The user moved wrong" closes real defects.** Defining a gesture is
  vocabulary; requiring stillness before it is asking the operator to supply a
  missing threshold.
- **A harness that picks its own inputs will pick the wrong ones.** `folds[0]`
  silently selected the one session guaranteed to fail, and the result looked
  exactly like the new code breaking a frozen acceptance. Name the session.
- **When the outcome cannot choose, the mechanism must.** Nineteen
  configurations passed both tests; only looking at what each did to the actual
  defect distinguished clearing it from getting away with it.
- **A third recording can rule out a survivor a sweep could only fail to
  distinguish.** The independent session did not just repeat the pass/fail
  count — its tightest episode specifically fell in the gap between `K = 2`
  and `K = 3`, which is what actually froze the choice.
- **State the margin, not just the verdict.** Three counts of separation is
  recorded next to the freeze, not smoothed into "validated," because the
  next session that measures it might come back different.
