# music_data.py
#
# Builds the music training data from Bach's 371 chorales (music21's corpus,
# approved by Eric 2026-11-02). Each chorale is also shifted into the other
# keys (-5..+6 semitones): still only Bach's notes.
#
# Chorales are split by piece BEFORE transposing, so the held-out pieces are
# never seen in any key during training.
#
# Run: .venv/bin/python music_data.py
# Output: music_data/train.json, train_groups.json, val.json, val_pieces.json

import json
import random
from pathlib import Path

from music21 import corpus

from music_tokens import score_to_grid, grid_to_ids

OUT_DIR = Path("music_data")
VAL_PIECES = 20           # chorales held out for checking
SEED = 0

grids = []
skipped = 0
for path in corpus.chorales.Iterator(returnType="filename"):
    score = corpus.parse(path)
    grid = score_to_grid(score)
    if grid is None:
        skipped += 1
        continue
    grids.append((str(path), grid))

random.Random(SEED).shuffle(grids)
val, train = grids[:VAL_PIECES], grids[VAL_PIECES:]

def encode_all(items, transpose):
    """Returns the sequences and, for each one, the number of its piece."""
    sequences, pieces = [], []
    for n, (_, grid) in enumerate(items):
        shifts = range(-5, 7) if transpose else [0]
        for shift in shifts:
            ids = grid_to_ids(grid, shift)
            if ids is not None:
                sequences.append(ids)
                pieces.append(n)
    return sequences, pieces

train_seqs, train_groups = encode_all(train, transpose=True)
val_seqs, _ = encode_all(val, transpose=False)

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "train.json").write_text(json.dumps(train_seqs))
(OUT_DIR / "train_groups.json").write_text(json.dumps(train_groups))   # piece number per sequence
(OUT_DIR / "val.json").write_text(json.dumps(val_seqs))
(OUT_DIR / "val_pieces.json").write_text(json.dumps([p for p, _ in val], indent=1))

print(f"chorales used: {len(grids)} (skipped {skipped} not four-voice)")
print(f"train: {len(train)} pieces -> {len(train_seqs)} sequences, "
      f"{sum(map(len, train_seqs)):,} tokens")
print(f"val:   {len(val)} pieces, {sum(map(len, val_seqs)):,} tokens")
