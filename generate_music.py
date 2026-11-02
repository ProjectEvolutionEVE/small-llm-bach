# generate_music.py
#
# Gives the trained model the opening bars of a chorale it never saw in
# training, lets it predict what follows, and writes:
#   output/<name>_model.wav / .mid / .musicxml   the opening + the model's continuation
#   output/<name>_bach.wav  / .mid / .musicxml   the same opening + Bach's real continuation
#   output/<name>_*_tokens.json                  the exact tokens behind each file
#
# Run: .venv/bin/python generate_music.py [piece_number] [prompt_bars] [new_bars]
#   piece_number: 0-19, one of the held-out chorales (music_data/val_pieces.json)

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from music21 import clef, corpus, expressions, key, meter, note, stream, tie
from transformers import GPT2LMHeadModel

from music_tokens import (
    BAR_ID, EOS_ID, STEPS_PER_QUARTER, VOICES,
    score_to_grid, grid_to_ids, ids_to_notes,
)

PIECE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PROMPT_BARS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NEW_BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

MODEL_DIR = "bachLLM"
CONTEXT = 384          # tokens of history the model sees (its limit is 512)
TEMPERATURE = 0.9
TEMPO_BPM = 72
SEED = 0

OUT_DIR = Path("output")
SAMPLE_RATE = 22050


# -----------------------
# Prediction
# -----------------------
def continue_piece(model, ids, new_bars):
    """Extend `ids` by `new_bars` bars, a chunk at a time, sliding the
    window so it never goes past what the model can see."""
    ids = list(ids)
    bars_made = 0
    while bars_made < new_bars:
        window = torch.tensor([ids[-CONTEXT:]])
        out = model.generate(
            window,
            attention_mask=torch.ones_like(window),
            max_new_tokens=64,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=EOS_ID,
        )[0, window.shape[1]:].tolist()
        for tok in out:
            if tok == EOS_ID:
                return ids + [EOS_ID]
            if tok == BAR_ID:
                bars_made += 1
                if bars_made == new_bars:
                    return ids
            ids.append(tok)
    return ids


# -----------------------
# Output files
# -----------------------
def write_midi(notes, path):
    score = stream.Score()
    for voice in notes:
        part = stream.Part()
        for start, length, midi in voice:
            n = note.Note(midi, quarterLength=length / STEPS_PER_QUARTER)
            part.insert(start / STEPS_PER_QUARTER, n)
        score.insert(0, part)
    score.write("midi", fp=str(path))


def time_signature(steps):
    if steps % 4 == 0:
        return f"{steps // 4}/4"
    if steps % 2 == 0:
        return f"{steps // 2}/8"
    return f"{steps}/16"


def write_score(ids, prompt_steps, sharps, label, color, path):
    """Write a MusicXML score with the bar lines exactly where the tokens
    put them. Notes after the opening are coloured, and a text marks where
    the continuation starts."""
    notes, bars, total = ids_to_notes(ids, with_bars=True)
    bounds = bars + [total]
    pickup = bounds[1] - bounds[0] < 16
    clefs = [clef.TrebleClef, clef.TrebleClef, clef.Treble8vbClef, clef.BassClef]
    names = ["Soprano", "Alto", "Tenor", "Bass"]

    score = stream.Score()
    for v, voice in enumerate(notes):
        part = stream.Part()
        part.partName = names[v]
        last_ts = None
        for b, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            if end <= start:
                continue
            m = stream.Measure(number=b if pickup else b + 1)
            if b == 0:
                m.insert(0, clefs[v]())
                m.insert(0, key.KeySignature(sharps))
            ts = "4/4" if (b == 0 and pickup) else time_signature(end - start)
            if ts != last_ts:
                m.insert(0, meter.TimeSignature(ts))
                last_ts = ts
            if b == 0 and pickup:
                m.paddingLeft = (16 - (end - start)) / STEPS_PER_QUARTER
            if v == 0 and start == prompt_steps:
                m.insert(0, expressions.TextExpression(label))

            t = start
            for s, l, midi in voice:
                a, z = max(s, start), min(s + l, end)
                if z <= a:
                    continue
                if a > t:
                    m.append(note.Rest(quarterLength=(a - t) / STEPS_PER_QUARTER))
                n = note.Note(midi, quarterLength=(z - a) / STEPS_PER_QUARTER)
                if s < start and s + l > end:
                    n.tie = tie.Tie("continue")
                elif s < start:
                    n.tie = tie.Tie("stop")
                elif s + l > end:
                    n.tie = tie.Tie("start")
                if s >= prompt_steps:
                    n.style.color = color
                m.append(n)
                t = z
            if t < end:
                m.append(note.Rest(quarterLength=(end - t) / STEPS_PER_QUARTER))
            part.append(m)
        score.insert(0, part)
    score.write("musicxml", fp=str(path))


def write_wav(notes, path):
    step_sec = 60 / TEMPO_BPM / STEPS_PER_QUARTER
    end = max((s + l for v in notes for s, l, _ in v), default=0)
    audio = np.zeros(int((end * step_sec + 1.0) * SAMPLE_RATE))
    for voice in notes:
        for start, length, midi in voice:
            freq = 440 * 2 ** ((midi - 69) / 12)
            n = int(length * step_sec * SAMPLE_RATE)
            t = np.arange(n) / SAMPLE_RATE
            # organ-like tone: a few harmonics, short fade in and out
            tone = (np.sin(2 * np.pi * freq * t)
                    + 0.4 * np.sin(4 * np.pi * freq * t)
                    + 0.2 * np.sin(6 * np.pi * freq * t))
            fade = min(n // 2, int(0.02 * SAMPLE_RATE))
            env = np.ones(n)
            if fade:
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
            i = int(start * step_sec * SAMPLE_RATE)
            audio[i:i + n] += tone * env
    audio /= max(1e-9, np.abs(audio).max()) / 0.8
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())


# -----------------------
# Main
# -----------------------
val_pieces = json.load(open("music_data/val_pieces.json"))
path = val_pieces[PIECE]
name = Path(path).stem
original = corpus.parse(path)
grid = score_to_grid(original)
key_sig = original.recurse().getElementsByClass("KeySignature").first()
sharps = key_sig.sharps if key_sig is not None else 0
prompt_steps = sum(len(b) for b in grid[:PROMPT_BARS])

prompt = grid_to_ids(grid[:PROMPT_BARS])[:-1]                     # drop EOS
real = grid_to_ids(grid[:PROMPT_BARS + NEW_BARS])

torch.manual_seed(SEED)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).eval()
predicted = continue_piece(model, prompt, NEW_BARS)

OUT_DIR.mkdir(exist_ok=True)
for label, ids, text, color in (
    ("model", predicted, "Small LLM predicts from here", "#1f6feb"),
    ("bach", real, "Bach's real continuation from here", "#1a7f37"),
):
    notes = ids_to_notes(ids)
    write_midi(notes, OUT_DIR / f"{name}_{label}.mid")
    write_wav(notes, OUT_DIR / f"{name}_{label}.wav")
    write_score(ids, prompt_steps, sharps, text, color, OUT_DIR / f"{name}_{label}.musicxml")
    (OUT_DIR / f"{name}_{label}_tokens.json").write_text(json.dumps(ids))

prompt_sec = prompt_steps * 60 / TEMPO_BPM / STEPS_PER_QUARTER
print(f"piece: {name} (never seen in training)")
print(f"opening: {PROMPT_BARS} bars of Bach, then the model's {NEW_BARS} bars "
      f"start at {prompt_sec:.1f} s")
print(f"wrote {OUT_DIR}/{name}_model.wav and {OUT_DIR}/{name}_bach.wav (+ .mid, .musicxml)")
