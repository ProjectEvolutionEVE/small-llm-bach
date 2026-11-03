# Experiment 2: predictions (written before any test runs)

Committed to git before training or scoring, so the timestamp shows these came first.
Results go in a separate file afterwards; this file is not edited once committed.

## What we found already (probe of the first model, `probe_pitch.py`)

Nobody told the first model how notes relate. Looking at the vectors it learned for each pitch
(`output/pitch_probe_baseline.txt`):

- It found the **octave** on its own: pitches 12 half-steps apart are far more alike than 11 or 13.
  Two octaves (24) shows the same, weaker.
- It partly found the **fifth** (7 half-steps), and set the **tritone** (6), the most clashing
  interval, apart from its neighbours.
- Still unresolved: a note's octave partner ranks only about 8th most alike of 45, and the
  **thirds and sixths** (5:4, 5:3) show almost no peak.

## The question

Does giving each token facts that physics already fixes help it learn Bach faster and better,
without telling it anything about Bach or music theory?

## Models (same size, same data, same training time, same 20 held-out chorales)

| Model | What it gets on top of the token itself |
|---|---|
| **M0 baseline** | Nothing (the first model's setup, retrained with the same script for a fair comparison) |
| **M1 position** | Where it is: which voice it is writing (S, A, T, B) and which sixteenth of the bar. Counting only. |
| **M2 physics** | For each pitch token: note name (pitch class), octave, frequency (log Hz), and where its first 8 overtones land. Physics of sound only, approved by Eric 2026-11-02. |
| **M3 both** | M1 + M2 |
| **M0-half, M2-half** | M0 and M2 trained on half the training chorales, to test "learns faster from less Bach" |

The attributes are fixed numbers added to each token. The model still has to work out what to do
with them. The token stream, the vocabulary and the held-out pieces are identical for every model,
so the scores compare directly.

## How we score

On all 20 held-out chorales:

- **Held-out loss**: how surprised the model is by Bach's real next token (lower is better).
- **Generated music**: 4 bars of Bach in, 8 bars out, 3 seeds per piece (60 continuations per model),
  compared with Bach's real 8 bars:
  - **Bar accuracy**: share of generated bars with the right number of beats.
  - **Voice order**: share of steps where a lower voice sits above a higher one.
  - **Clashes**: share of steps where two voices sound a half-step or major 7th apart.
  - **In key**: share of notes in the chorale's scale (key found by music21, used only for scoring;
    the model never sees it).
  - **Complexity**: notes started per bar, and distinct chords (sets of note names) per bar.

## Predictions

### Claude

1. **Baseline (M0)**: bar accuracy between 40% and 70%. Clash rate above Bach's.
2. **Position (M1)**: bar accuracy rises above 90%; voice-order errors fall by at least half; held-out
   loss improves by at least 0.03.
3. **Physics (M2)**: held-out loss improves only a little (0 to 0.03). Clashes fall 10–30% versus
   M0. Bar accuracy about the same as M0. Inside the model, octave partners rank in the top 3 and a
   peak appears at the thirds.
4. **Half data**: M0-half is clearly worse than M0 (loss at least 0.05 higher). M2-half closes at
   least half of that gap. This is where physics should help most.
5. **Both (M3)**: best on every score, but still clashes more than Bach.
6. **More complex than Bach**: about 5% chance (definition below).

### Eric

> "Takes what you said and also includes the possibilities of taking Bach into an even more
> dynamic and complex composition. I give this roughly a **15% chance** of adding an additional
> layer of complex scores and arrangements to the composition than the original Bach."

### Measuring "more complex than Bach" (proposed, Eric may amend before results)

Counted as a yes if a physics model (M2 or M3), averaged over its 60 continuations, has **more
notes started per bar and more distinct chords per bar than Bach's real continuations**, while
staying clean: **bar accuracy at least 90% and clashes no higher than Bach's**. Busier but messier
does not count.
