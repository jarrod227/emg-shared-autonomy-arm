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

## What is left: one class does not survive redonning

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

The consequence is that recording one more event-gate session does not close
this out. It may land in a third position. Training data has to span several
genuinely separate donnings before cross-session accuracy means anything, and
the gesture has to be defined tightly enough to be repeatable — the current
definition is the two words `WRIST UP`.

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
