# Small LLM: a model that learned Bach from nothing

A small GPT-2-style model trained **from scratch** on J. S. Bach's four-part chorales, then asked to
continue a chorale it never saw. No internet, no pretrained weights, no outside data except the
chorales. It trains in about 15 minutes on a CPU.

Full write-up with audio and scores: **[Lab notes on risesolutions-llc.com](https://risesolutions-llc.com/lab/small-llm/)**

## First trial run

Opening 4 bars of Bach's chorale from Cantata BWV 4 (held out from training), then 8 bars predicted
by the model. Temperature 0.9, seed 0, first and only run, not cherry-picked.

| | |
|---|---|
| Model's version | [`output/bwv4_model.wav`](output/bwv4_model.wav) |
| Bach's real continuation | [`output/bwv4_bach.wav`](output/bwv4_bach.wav) |
| Both scores side by side | [`output/bwv4_compare.html`](output/bwv4_compare.html) |
| All 75 tokens with counts | [`output/token_catalog.html`](output/token_catalog.html) |

**What it learned:** it stays close to the key, keeps the voices in range and moves them like a hymn.
**What it got wrong:** it miscounts beats (bars of 7/8, 9/8 and 5/8), so notes sometimes slip into
the wrong voice, and its phrases rarely come to rest.

## Where it started

This project began with Hassan Habib's
[LLMfromNothing](https://github.com/hassanhabib/LLMfromNothing): three short scripts that train a
GPT-2-style model from random numbers on a tiny `User:` / `Assistant:` text file, test it, and
convert it for llama.cpp. They gave us a working pipeline on day one, and everything here grew from
them. Thank you, Hassan.

His repo has no license, so his scripts are not included here. Before running them we made these
corrections:

- **Prompt mismatch.** Training split each example into `User: Hello` and `Assistant: World!`, so the
  model learned to write `Assistant:` itself. The test prompt already ended in `Assistant:`, so the
  model would have written it a second time. Fix: train on the same `User: …\nAssistant:` prompt the
  test uses, and learn only the reply. After the fix it answered once and stopped.
- **Comment vs data.** A comment said the data used no roles, while the data file has them. The new
  parser reads the `User:` / `Assistant:` roles and skips any block missing either.
- **Our machine (not a flaw in his code).** Intel Macs stop at PyTorch 2.2.2 on Python 3.12, so the
  other libraries are pinned to match (see `requirements.txt`).

The music version here is new code built on his idea: a home-made note vocabulary in place of the
text tokenizer, and Bach's chorales in place of the text file.

## How notes become tokens

Each bar is cut into sixteenth-note steps. Every step is 4 tokens, one per voice
(soprano, alto, tenor, bass):

- `P60`: a note starts (MIDI pitch 60 = middle C)
- `HOLD`: the voice keeps sounding its note
- `REST`: the voice is silent
- `BAR`: bar line (takes no time). `BOS` / `EOS`: start and end of a piece

That gives 75 tokens in all, built here (no downloaded vocabulary). The model's only task is to
predict the next token. The real next token in Bach's score is the answer key.

## Model and data

- GPT-2 architecture, 6 layers, 384 wide, 6 heads, 10.9M parameters, random start
- 351 four-voice chorales from the [music21](https://www.music21.org/) corpus. 20 are held out
  before training, and the other 331 are shifted into all 12 keys, giving 3.47M training tokens.
- 2 epochs. Held-out loss 0.63 vs training loss 0.67.

## Run it

Tested on an Intel Mac with Python 3.12 (PyTorch 2.2.2 is the last version for Intel Macs, so the
other packages are pinned to match).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python music_data.py        # build tokens from the chorales (~1 min)
.venv/bin/python train_music.py       # train (~15 min on CPU) -> bachLLM/
.venv/bin/python generate_music.py 0 4 8   # held-out piece 0, 4 bars in, 8 bars out -> output/
.venv/bin/python score_compare.py     # side-by-side scores -> output/<name>_compare.html
.venv/bin/python token_catalog.py     # token list -> output/token_catalog.html
```

## Files

| File | What it does |
|---|---|
| `music_tokens.py` | The note vocabulary and encode/decode |
| `music_data.py` | Builds training data from the chorales |
| `train_music.py` | Trains the model |
| `generate_music.py` | Continues a held-out chorale, writes WAV, MIDI, MusicXML |
| `score_compare.py` | Draws both scores side by side (Verovio) |
| `token_catalog.py` | Lists every token with its training count |

## Credits

Started from Hassan Habib's [LLMfromNothing](https://github.com/hassanhabib/LLMfromNothing).
Chorales by J. S. Bach (public domain), via the music21 corpus. Built with PyTorch and Hugging Face
Transformers. Scores engraved with [Verovio](https://www.verovio.org/).

© 2026 Rise Solutions LLC
