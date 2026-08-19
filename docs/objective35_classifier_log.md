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

## The day the threshold's own assumption failed (2026-08-15)

The bridge's first live acceptance rounds surfaced three separate findings in
one morning, each of a different kind: a parameter estimated instead of
measured, a hardware repair that broke a software protection, and a design
assumption that two donnings of data finally falsified.

### The confirmation window was estimated, and sat below reality

The bridge requires two complete MCU events within a window before
publishing `NEXT_TARGET` or `CONFIRM`. The window shipped at 3.0 s, derived
from MCU timing — hold-off plus stable run plus re-arm. A live round then
measured the actual cadence: consecutive deliberate events arrived 3.80,
3.95, and 4.15 s apart, because the human relax time between gestures is
part of the interval and was never in the derivation. Every pair missed the
window; the bridge published nothing at all.

The replacement was measured from both sides. The floor is 4.15 s, the
slowest deliberate gap. The ceiling is 7.05 s, the closest two same-command
events in the ten-minute ordinary-activity recording — and that pair
deserves its footnote: at any window of 7 s or more it is held apart only by
an `ABORT` landing 0.05 s later and clearing the pending half, which is
passing on interference, not margin. 5.5 s sits 1.35 s above the floor and
1.55 s below the ceiling.

### A repaired electrode broke the activation threshold

A silent acceptance round — link healthy, quality 255, zero events over five
gestures — led back to the scope. The signal had degraded across the
redonning: rest at 32 total MAV but gestures at only 98–143 against the
3 × 32 = 96 threshold, with `ABORT` at 98 clearing it by 2 counts and never
accumulating five agreeing windows. The cause was one electrode: ch0 rested
at MAV 19 with ±100 spikes where ch2 rested at 3. Re-gelling it fixed the
noise — and exposed the design flaw.

With contact repaired, rest fell to a drifting 6–43. The threshold followed
it down to as little as 3 × 6 = 18, below the measured 73-count preparatory
movement the activation stage exists to suppress. Meanwhile the gesture band
barely moved. Laid side by side, the two donnings falsify the scaling rule
itself:

| | rest | preparation | weakest gesture | needed K |
| --- | --- | --- | --- | --- |
| 2026-08-14 donning | 29–35 | 65–79 | 318 | ≈ 3 |
| 2026-08-15, re-gelled | 6–43 | 73 | 145 | 2.5–18 |

Rest amplitude is a contact-noise figure; the preparation/gesture band is
set by physiology and electrode placement. The two do not covary, so no
single multiple of rest can place a threshold between preparation and
gesture across donnings. K = 3 worked on 2026-08-14 because that donning's
noise happened to put 3 × rest inside the band — a coincidence that one
electrode repair undid.

### An interim floor, and the measured fix it stands in for

The stopgap is an absolute floor under the relative rule:
`threshold = max(K × baseline, 110)`, with 110 sitting 31 counts above the
loudest measured preparation (79) and 35 below the weakest measured gesture
(145). It also closes a cold-start gap: before the first classified-REST
window the stage used to pass everything unjudged, and now suppresses
sub-floor movement while keeping a forceful cold-start `ABORT` reachable.
Replaying all three existing recordings — the defect session, the frozen 6/6
acceptance, and the ten-minute ordinary-activity run — with and without the
floor produced identical events on all three, and the reflashed firmware
passed a live round: `NEXT_TARGET` paired and published, `CONFIRM` paired
and published, `ABORT` fired single-shot with margin 20 — the first
successful `ABORT` of the day. The floor is recorded as interim on two
donnings of evidence, not frozen.

The real fix, agreed in design: per-donning calibration. Measure rest and a
few comfortable real gestures at each donning; take the preparation band
from the onset segments of those same trials; place `T_session` midway
between the preparation upper bound and the weakest-gesture lower bound; and
make the separation ratio itself the acceptance criterion — today's donning
separates by 2.0 (73 to 145) where yesterday's separated by 4.0, and a
calibration that measures 2.0 should fail loudly and ask for re-placement
rather than proceed on a sliver. One candidate was evaluated and cut:
normalizing by comfortable effort, `(mav − rest) / (effort − rest)`, does
not remove the guessed constant, because comfortable effort itself swung
3.7× between donnings (736 vs 197) — the normalized preparation of one
donning (0.351) nearly reaches the normalized weakest gesture of the other
(0.404). Anchoring on the measured gap between preparation and gesture keeps
both bounds inside the same donning. Delivering `T_session` to the MCU
requires a host-to-device configuration path the wire protocol does not yet
have; that is the next milestone, and the same downlink later carries the
proportional-control calibration.

### A comparison tool that expires 72 minutes after power-on

Verifying the floor against the new donning's recording tripped a latent
bug in `emg_runtime_compare.py`: it rejects the capture with "RAW timestamp
is not on the device frame grid." The timestamps are fine. The wire
timestamp is uint32 microseconds and wraps every 71.6 minutes, and
2³² mod 500 = 296, so each wrap shifts the timestamp residue by 296 µs. The
tool checks `timestamp % 500 == 0` as an absolute property, which only holds
during the first wrap period after MCU reset. Every recording taken from a
board powered longer than 71.6 minutes fails the check; the recording in
question was taken about 3.7 hours in (residue 112 = three wraps). The fix —
deriving the grid phase from the stream itself instead of assuming residue
zero, and unwrapping across in-file wraps — is queued; until then the tool
quietly imposes a "freshly reset boards only" precondition it never states.

## A calibration downlink, and three failed calibrations that were the tool's fault

The protocol grew its first host-to-device packet the same day: `SET_ACTIVATION`
(host to device, apply values or restore compile-time defaults) and
`ACTIVATION_STATE` (device to host, a state report rather than an ACK, so a
lost reply just means the host keeps watching rather than guessing whether to
resend). No version bump — `0x80`-`0xFF` was reserved for this direction from
the protocol's first draft. Firmware gained a lock-free single-producer/
single-consumer ring so the USB interrupt callback can hand bytes to the main
loop without parsing in interrupt context, and `emg_activation_reconfigure`
changes K/shift/floor on a running instance without losing the measured
baseline, rescaling the accumulator on a shift change so the baseline cannot
silently move by a power of two. Verified live end to end before any real
calibration was attempted: apply, reject-without-mutating (a hand-built
out-of-range request left the previous configuration provably untouched), and
restore-defaults all confirmed via `ACTIVATION_STATE` with zero packet loss.

`emg_calibrate.py` then measures one donning's rest, preparation, and gesture
bands and computes `T_session` as the geometric mean of the preparation upper
bound and the weakest gesture's sustained level, gated on a three-tier
separation ratio (`>= 3.0` pass, `2.5-3.0` marginal, `< 2.5` fail). The first
three real attempts, all on 2026-08-15, all failed: separation 1.52, then
1.40 despite better contact and a more realistic preparation prompt, with
`ABORT` sustaining a reported 98-130 against a preparation of 70-86. That
pattern — the metric getting worse while everything observable about the
donning got better — was the signal something was wrong with the tool, not
the arm.

A scope trace of the same `ABORT` motion made it undeniable: a clean
contraction reaching roughly ±500-600 counts on two channels, occupying the
back 28% of a 3-second capture window. Reconstructing the actual held level
from that trace gave about 536 total MAV against a reported 98 — a 5.5x
error. The plateau estimator was the 75th percentile of the whole trial. With
a second or so of reaction time before the hold started, the top quartile of
windows landed on the onset ramp, not the hold; preparation used the 95th
percentile, which *does* catch a brief peak, so the two errors moved in
opposite directions and compounded. Three consecutive real failures, and the
hardware had been fine throughout.

The fix replaces both percentiles with one rule: the highest level sustained
for at least as many consecutive windows as the event gate requires before it
will fire anything (`VALIDATED_GATE.stable_windows`, imported rather than
restated so the two cannot drift apart). It is insensitive to where in the
trial the hold happened and states the correct thing about preparation for
free — a movement too brief to fill the gate's stable run cannot fire an
event at any threshold, so it must not be allowed to drag the threshold down.
Gesture and preparation prompts were also lengthened from 3 s to 4 s for
reaction time, and the tool now stores the raw per-window arrays in the
output JSON — the first version stored only the summary, and the bug that
produced the 5.5x error was actually found from a screenshot, because there
was nothing left in the file to re-derive it from.

A same-session, same-electrodes repeat with the fixed estimator scored
separation 3.24, `ABORT` sustaining 295 against a preparation of 91, and
passed. The board confirmed `K=3 shift=4 floor=164` via `ACTIVATION_STATE`.

### Live acceptance, and a hold-duration floor nobody had stated

`CONFIRM` and `ABORT` both fired correctly on the bridge's first attempt at
each. `NEXT_TARGET` did not, six times in a row, despite a directly measured
sustained level (208 against the 164 floor) that should have cleared it. The
first diagnostic script explained nothing new; a second, which replays the
recorded samples through the real filter, classifier, activation, and gate
in sequence rather than trusting any single stage's summary, did: the
classifier called every loud window `NEXT_TARGET` correctly and the
activation stage passed them, but each attempt held the contraction for only
about 15-17 feature windows, and the gate needs `hold-off(12) + stable(5) =
17` consecutive windows past the moment it leaves REST before it will emit
anything — a floor of roughly **0.85 s**, in exactly the position the wire
protocol reports no window-by-window classifier state, so a wearer failing
against it sees nothing but silence. A seventh attempt, held deliberately
longer, fired at window 61 exactly on schedule. Amplitude was never the
problem; every failure was a hold shorter than the gate's own requirement,
and nothing before this session had ever measured or stated that number.

Lowering the floor was considered and rejected: the trace showed comfortable
margin above threshold throughout every failed attempt, so a lower floor
would not have helped duration and would have eroded exactly the amplitude
margin the floor exists to protect. Shortening the 12-window onset hold-off
was also considered — it was frozen before the activation stage existed, and
the activation stage now independently removes the low-amplitude ramp the
hold-off was originally built to cover, so the two may be paying the same
cost twice. Rejected for today on the same principle as the floor: it is a
frozen, evidence-backed safety parameter, and the fix for an unstated latency
requirement is to state it, not to spend margin nobody has re-measured. Left
as an open question for a dedicated sweep against the existing acceptance
recordings, now that the activation stage is part of the pipeline being
swept.

One more finding, unrelated to any bug: showering removes the skin's own
conductive layer (sebum, salts), and the wear-detect lines read the
resulting high impedance as no contact — reproduced on ch2 within the same
session. Touching the electrode with a slightly oily fingertip restored
contact immediately. Not logged as a defect; logged as a real, reproducible
donning variable a wearer should know about, alongside electrode placement
and gel condition.

### The ordinary-activity false-trigger check, re-run under today's config

Today changed three things that the earlier 10-minute ordinary-activity
false-trigger result (2026-08-14, `/tmp/emg_ordinary_10min.bin`) never saw
together: a per-session calibrated floor (164, not the compile-time default
110), a measured confirmation window (5.5 s, not the original 3.0 s
estimate), and the startup handshake gate. Re-collecting ten more unscripted
minutes was unnecessary — the recorded RAW samples are unaffected by any of
it — so the same file was replayed through the exact pipeline in production
today instead.

One subtlety in doing this correctly: `emg_runtime_compare.replay_host`
constructs `ActivationGate()` with no arguments, and that class's defaults
are bound once, when `emg_activation_ref.py`'s class body first executes —
poking `emg_activation_ref.THRESHOLD_FLOOR` afterward does not reach it,
because Python binds default-argument values at `def` time, not at call
time. The correct injection point is the name `ActivationGate` as looked up
inside `emg_runtime_compare`'s own module namespace at call time, which
*does* follow a rebinding. This matters retroactively: an earlier same-day
floor regression check used the constants-poking approach and reported
"identical events with and without the floor" on all three prior recordings
— a result that is still probably true (three independent live-hardware
acceptances after it agree), but was not actually proven by that script,
since both of its runs may silently have used the same real default rather
than the two floors being compared.

Result, replayed through today's actual floor (164) and window (5.5 s): 4
MCU-equivalent candidate events over ten minutes (2 `ABORT`, 1
`NEXT_TARGET`, 1 `CONFIRM`), of which 2 reached `/assistive_intent` — both
`ABORT`, the safe-side direction. Neither the `NEXT_TARGET` nor the
`CONFIRM` candidate found a matching second event inside the window, so
neither published; this is the double-event policy working as designed, not
an absence of signal. Zero events in the hazardous direction, and fewer
total candidate events than the original floor=110 check, which is the
expected direction for a higher, better-measured floor.

## Choosing a direction gesture, and finding it cannot carry an amplitude (2026-08-18)

Continuous view control needs `LEFT` and `RIGHT`, and the four trained intents
have no spare gesture: `ABORT` is globally reserved, `CONFIRM` means lock and
approach. Three candidates were recorded alongside the existing four in one
donning, two sessions, three repetitions each: radial deviation, ulnar
deviation, and supination. Pronation was dropped before recording -- the
forearm rests palm-down, which is already full pronation, so the class would
have had no range.

### The comparison that decided it

Seven classes together reached 95.2% trial and 85.0% window accuracy, but that
figure decides nothing. `ABORT` is the safety class and was already the
weakest of the four, so what matters is what each candidate costs it. The
baseline has to be a four-class model trained on *this* dataset, not the 96%
from the 2026-08-14 sessions -- that was a different donning with five
sessions, and comparing across it would attribute donning differences to the
candidates.

| model | window | trial | `ABORT` window | vs base | candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| four-class baseline | 94.3% | 100.0% | 86.1% | | |
| + radial | 93.9% | 100.0% | 86.1% | +0.0pp | 92.4% |
| + supination | 87.5% | 100.0% | 86.1% | +0.0pp | 71.7% |
| + ulnar | 91.0% | 100.0% | 85.8% | -0.3pp | 80.2% |
| + radial + ulnar | 90.8% | 100.0% | 84.9% | -1.2pp | |
| all seven | 85.0% | 95.2% | 84.6% | -1.4pp | |

The anatomical prediction made before recording was wrong on the ordering. It
expected forearm rotation to sit farthest from `ABORT` in feature space, warned
that radial deviation would collide with `NEXT_TARGET`, and called ulnar
deviation the dangerous one because flexor carpi ulnaris is a flexor like the
one `ABORT` uses. Radial deviation turned out best, ulnar deviation did not
touch `ABORT` at all, and only the supination prediction held -- it is the
weakest, plausibly because the supinator is deep and biceps sits above the
electrode band. The prediction was recorded as reasoning rather than
measurement, and measuring was the point.

Both directions as a pair beat one direction plus a reused `NEXT_TARGET`, for
a reason no accuracy column shows: the pair frees `NEXT_TARGET` entirely, so
one gesture stops carrying two meanings. Their cross-confusion -- the failure
that matters most, since it means moving the wrong way -- was zero within the
donning, in both directions, at both window and trial level.

### Re-donning, which is where the real result is

Leave-one-session-out inside one donning measures repeatability, not transfer;
the electrodes never moved. A second donning, recorded the same evening with
the same protocol, allows training on one and testing on the other.

| fold | model | window | trial | `ABORT` | radial | ulnar |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A trains, B tests | four-class | 99.0% | 100.0% | 96.2% | | |
| A trains, B tests | + both | 89.8% | 94.4% | 95.6% | 77.0% | 86.3% |
| B trains, A tests | four-class | 94.3% | 100.0% | 84.9% | | |
| B trains, A tests | + both | 88.0% | 97.2% | 83.5% | 92.7% | 79.9% |

`ABORT` survives the re-donning, losing 0.6 and 1.4 points. The directions
never confuse each other at trial level in either fold, but the within-donning
"literally zero" does degrade: one fold called 26 of 344 ulnar windows radial.
The gate's five-window agreement and the majority vote absorbed all of it,
which is what the trial-level zero is made of -- so the claim is "zero at trial
level", not "zero".

The cross-donning confusion also produced what the anatomical prediction had
expected and the within-donning result had denied: 19% of `NEXT_TARGET` windows
were called radial, and 23% of radial windows were called `REST`. Radial
deviation and wrist extension are not as separable across donnings as within
one.

### The finding that overturned the choice

Classification accuracy was the wrong acceptance criterion, and picking a
winner on it repeated the mistake this project already has a lesson about.
The event gate does not act on windows, it acts on a level *sustained* across
five consecutive windows, and it never sees anything the activation threshold
has rewritten to `REST`. Measured that way:

| donning | `T_session` | preparation | `NEXT_TARGET` | radial | ulnar |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 123 | 59 | 145 | **84** | 116 |
| B | 116 | 64 | 171 | **93** | 180 |

**Radial deviation does not sustain above the activation threshold in either
donning.** It has peaks above it -- which is why every window above 116
classified correctly, 100% in every band -- but it cannot hold five
consecutive windows there, so the gate never fires. Ulnar deviation is -7 in
one donning and +64 in the other.

Lowering the threshold does not rescue it, and the reason is not the risk the
existing warning names. Two thresholds, a lower one for the continuous channel
and the existing one for events, is a defensible design: the two paths have
genuinely different failure modes, since a marginal crossing produces one full
discrete command but only a near-zero angle on the continuous path, and the
controller's 0.05 activation deadband is a second independent gate that
marginal crossings cannot pass. The event path would keep its threshold, so
the preparatory-movement defect stays closed.

What defeats it is dynamic range. Placing a threshold at the geometric mean of
preparation and the gesture, as the calibration already does:

| donning | gesture | preparation | level | `T_view` | span | degrees per count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `NEXT_TARGET` | 59 | 145 | 92 | 53 | 1.7 |
| A | radial | 59 | 84 | 70 | **14** | **6.6** |
| A | ulnar | 59 | 116 | 83 | 33 | 2.7 |
| B | `NEXT_TARGET` | 64 | 171 | 105 | 66 | 1.4 |
| B | radial | 64 | 93 | 77 | **16** | **5.7** |
| B | ulnar | 64 | 180 | 107 | 73 | 1.2 |

Radial deviation leaves 14 to 16 counts to encode a 90 degree range, about six
degrees per count, against a hold-to-hold coefficient of variation measured at
36% and a hop-to-hop change of 6%. Its separation from preparation is 1.42 and
1.45, where the calibration's own acceptance criterion fails anything below
2.5 on the grounds that such a threshold "would sit within noise of real
gestures".

### What this leaves

Discrete stepping needs classification and nothing else -- no amplitude, no
reference, no second threshold. Radial and ulnar deviation classify well
enough for it, and `DiscreteViewSweep` with a per-session mode parameter was
already built. Proportional control on these gestures is blocked on the
wearer's dynamic range, not on any missing software.

Three donnings also gave reference-to-threshold ratios of 2.38, 3.34 and 2.47.
That 40% spread is the evidence that a firmware cannot derive a reference level
from the threshold it already holds, whenever a reference is needed again.

Unresolved, and worth stating rather than leaving to be rediscovered: whether
radial deviation is intrinsically this weak on a forearm band, or was simply
performed gently. Both donnings were recorded in one evening by one wearer who
had already completed several sessions, and no capture asked for a deliberately
forceful deviation. That measurement costs minutes and was not taken.

## Lessons

- **An accuracy figure is not an acceptance criterion.** Radial deviation won
  the candidate comparison on classification and cannot fire the event gate at
  all, because the gate acts on a level sustained across five windows and the
  activation threshold rewrites everything below it to `REST`. The winner was
  declared before checking the physical requirement the class has to meet.
- **Within-donning repeatability hides the confusions that matter.** The two
  directions had literally zero cross-confusion until the electrodes were
  removed and re-applied, and the radial/`NEXT_TARGET` collision that the
  anatomical prediction expected appeared only across donnings.
- **A capture that asks for an abstract quantity fails; one that names a
  gesture works.** The comfortable-effort reference returned two of three
  trials inside the preparation band on both donnings and with two different
  wordings, because the wearer could not tell what was being asked and did
  nothing. Every gesture prompt in the same tool worked first time.
- **Check whether a shared parameter can serve both configurations it is
  applied to.** One `view_step_angle` was fine for the unloaded search band
  and wider than the whole loaded one, which silently removed the
  straight-ahead angle from the loaded sweep.

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
- **A parameter derived from the machine's timing forgets the human's.** The
  3.0 s confirmation window accounted for hold-off, stable run, and re-arm,
  and omitted the second or two a person takes to relax between gestures.
  The result was not a degraded pass rate but zero output.
- **Fixing the hardware can break the software that compensated for it.**
  Re-gelling a noisy electrode cut resting amplitude five-fold and, with it,
  a threshold defined as a multiple of resting amplitude. An improvement in
  the signal is a change in the operating point, and anything calibrated
  against the old one has to be re-checked.
- **A ratio is only a rule if both its terms move together.** Rest is
  contact noise, the gesture band is physiology. K = 3 was never a law; it
  was one donning's coincidence, and it survived exactly until an electrode
  was repaired.
- **Reject the tidy normalization if its denominator is not repeatable.**
  Dividing by comfortable effort looks dimensionless and principled, but
  effort itself varied 3.7× between donnings, so it relocated the guessed
  constant instead of removing it.
- **An absolute check on a wrapping counter has a shelf life.** `timestamp %
  500 == 0` was true of every recording made so far and false of every
  recording made more than 71.6 minutes after reset. The precondition was
  real from the first commit and only became visible when a board stayed
  powered overnight.
- **A percentile is a plateau estimator only when the trial is mostly
  plateau.** A second of reaction time before a held gesture put the
  75th-percentile "plateau" on the onset ramp instead, misreporting a
  536-count `ABORT` as 98 and failing three real calibrations before a
  scope trace made the actual hardware undeniable. Measure the thing the
  consumer (the event gate) actually needs — a level sustained for as many
  windows as the gate requires — not a statistic that happens to coincide
  with it under a timing assumption nobody wrote down.
- **A patched module-level constant only reaches code that looks it up at
  call time, not code whose default argument already bound it.** Poking
  `emg_activation_ref.THRESHOLD_FLOOR` after import does nothing to
  `ActivationGate()` calls inside `emg_runtime_compare.replay_host`, whose
  default parameter values were fixed when the class body first executed.
  The fix is to patch the name the caller's module actually resolves at
  call time. An earlier same-day "floor makes no difference" regression
  check used the broken pattern; its conclusion happened to hold up against
  independent live evidence, but the check itself proved nothing.
- **Every existing recording is worth re-running before asking for a new
  one.** The 10-minute ordinary-activity false-trigger result depended on
  the activation floor and the confirmation window, both of which changed
  today. Replaying the same RAW samples through today's actual
  configuration answered the safety question without costing anyone ten
  more minutes.
