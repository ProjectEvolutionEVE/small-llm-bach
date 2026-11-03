# Experiment 2: results

Predictions were committed first, in [`PREDICTIONS.md`](PREDICTIONS.md). Everything here was measured
afterwards on the 20 held-out chorales, with 3 seeds each, giving 60 continuations per model. Raw
numbers are in `results/`.

## Scores

| Model | Held-out loss | Bar accuracy | Clashes | Voice-order errors | In key | Notes / bar | Chords / bar |
|---|---|---|---|---|---|---|---|
| **Bach (real)** | | 100% | 5.1% | 0.9% | 91% | 14.5 | 4.55 |
| M0 baseline | 0.622 | 39% | 16.8% | 2.7% | 84% | 14.5 | 4.45 |
| M1 position | 0.540 | **94%** | 10.8% | 2.9% | **88%** | 14.5 | 4.67 |
| M2 physics | 0.610 | 54% | 16.9% | 3.3% | 83% | 13.4 | 4.08 |
| M3 both | **0.534** | 92% | **10.5%** | 2.8% | 86% | 14.0 | 4.48 |
| M0-half | 1.032 | 2% | 31.5% | 68.9% | 81% | 13.4 | 5.83 |
| M2-half | 1.017 | 2% | 27.2% | 54.1% | 86% | 10.7 | 4.94 |

How sure we are: across the 20 pieces, the 95% range for bar accuracy is about ±7 points and for
clashes about ±3 points. Each model was trained once, so a different training run could shift
these further. Differences smaller than that are treated as no change.

## Looking inside (pitch probe)

| Model | Octave partner in top 3 | Peak at the minor 3rd | Peak at the major 3rd |
|---|---|---|---|
| M0 baseline | 22% of pitches | +0.10 | none (−0.05) |
| M2 physics | 100% | +0.11 | none (−0.04) |
| M3 both | 100% | +0.10 | none (−0.06) |

The octave result comes straight from the physics facts: an **untrained** physics model already
puts octave partners first. The model kept that link, but it didn't discover it. Physics added no
new peak at the thirds; the minor-3rd peak was already in the baseline.

## Predictions vs results

### Claude

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | M0 bar accuracy 40–70% | 39% | **Miss** (just under) |
| 1 | M0 clashes above Bach | 16.8% vs 5.1% | Hit |
| 2 | M1 bar accuracy above 90% | 94% | Hit |
| 2 | M1 voice-order errors halved | 2.7% → 2.9% | **Miss** (no change) |
| 2 | M1 loss improves at least 0.03 | −0.083 | Hit |
| 3 | M2 loss improves 0–0.03 | −0.013 | Hit |
| 3 | M2 clashes fall 10–30% | 16.8% → 16.9% | **Miss** (no change) |
| 3 | M2 bar accuracy about the same | 39% → 54% | **Miss**: physics helped more than expected |
| 3 | Octave partners in top 3 | 100% | Hit, but built in (see probe) |
| 3 | New peak at the thirds | none | **Miss** |
| 4 | M0-half loss at least 0.05 worse | +0.41 | Hit |
| 4 | M2-half closes at least half the gap | closes 0.015 of 0.41 (4%) | **Miss** |
| 5 | M3 best on every score | best loss and clashes; M1 better on bars, key, voice order | **Miss** |
| 5 | M3 still clashes more than Bach | 10.5% vs 5.1% | Hit |
| 6 | "More complex than Bach": ~5% | No | Consistent |

### Eric

"More complex than Bach" at ~15%: **didn't happen.** Neither physics model wrote more notes or
more chords per bar than Bach (M2 13.4 / 4.08, M3 14.0 / 4.48, Bach 14.5 / 4.55), and neither
was as clean as Bach (clashes 10–17% vs 5%). A single result can't settle a probability: 15% and
5% both expected "no".

The only model busier than Bach was M0-half, at 5.8 chords per bar. That came from disorder, not
complexity: 69% voice-order errors and 2% correct bars. This is the "busier but messier" case the
definition excludes.

## What we learned

1. **Counting the beat is the big win.** Telling the model where it is in the bar took correct
   bars from 39% to 94% and cut clashes by a third. That is the largest gain in the experiment.
2. **Physics helps a little and in a place we didn't predict.** It made the loss slightly better and
   the bar counting better (39% → 54%), but it did nothing for clashes. With position added, physics
   adds almost nothing on top (M3 vs M1).
3. **Physics doesn't rescue a model with too little Bach.** On half the data both models fail to
   keep four voices in order. Physics makes that failure a little less bad (fewer clashes and
   voice errors, more notes in key), but it doesn't close the gap.
4. **Still short of Bach.** The best models clash about twice as often as Bach, and in more than half
   of runs they end the piece before the 8 bars are up.

## Disclosures

- Two scoring rules were settled **before any results**, while testing the scorer on one piece
  with the original model:
  - A bar counts as correct if its length matches **any** bar length Bach uses in that piece.
    The first version accepted only the most common length, which scored Bach's own music at 56%
    because of his phrase-end bars and pickups.
  - Generation stops at the 9th bar line, so the model gets 8 full bars like Bach.
- Bach's real continuation averages 7.4 bars, because some held-out pieces end sooner.
- The models often end the piece early: 42% of M0's runs and 60–78% of the others' end before 8
  bars, and the scores cover only the bars that were written.
- The M1 probe is less reliable: the position facts it adds make every pitch look more alike, even
  untrained.
- The scoring code (`evaluate.py`) was first drafted by a smaller Claude model, then reviewed and
  corrected before use.

## Next test (proposed 2026-11-03, not run)

Eric's idea: each voice lives in its own octave range, so with physics facts and enough training
time the model should learn which voice comes next from pitch height. Test: M2-half trained for as
many steps as the full-data models (same time, half the Bach), compared with M2-half and M2.
Needs a step-count option in `train_music.py`. Commit predictions before running.

**Eric's prediction (recorded 2026-11-03 05:02, before anything ran):**

> "I do not know how many attempts it will take, but small gains will make a slow increase and not
> every one will beat the previous one, but like a rising stock, the slow trend would be an upward
> trajectory. Until, it finally gets the full picture after it self tests it's own theory. Then we
> will see a significantly improved success rate that jumps off the page. How many attempts? This I
> cannot predict."

To test it, save the model at regular points during training and score each one: a slow rise
followed by a sudden jump only shows up as a curve, not as a single final score.
