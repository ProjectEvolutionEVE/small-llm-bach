# probe_pitch.py
#
# Looks inside the trained Bach model (bachLLM/) to see what it worked out on
# its own about pitch. The model was never told that P60 and P72 are the same
# note an octave apart, or how far apart any two notes are. Each pitch token
# got a random vector at the start; training moved those vectors around.
# If two pitches end up with similar vectors, the model treats them as alike.
#
# For every pair of pitches we measure how similar their vectors are
# (cosine similarity, -1 to 1) and average by the distance between them in
# half-steps. The same is done for an untrained model with random vectors,
# which shows what "nothing learned" looks like.
#
# Run: .venv/bin/python probe_pitch.py

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from music_tokens import TOKEN_TO_ID

LOW, HIGH = 36, 81   # pitches common in the training data (C2 to A5)
NAMES = {1: "half-step", 2: "whole step", 3: "minor 3rd", 4: "major 3rd",
         5: "4th", 6: "tritone", 7: "5th", 8: "minor 6th", 9: "major 6th",
         10: "minor 7th", 11: "major 7th", 12: "OCTAVE", 19: "octave + 5th",
         24: "2 OCTAVES"}


def by_interval(vectors):
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = v @ v.T
    n = len(v)
    return {d: float(np.mean([sim[i, i + d] for i in range(n - d)]))
            for d in range(1, 26)}


def pitch_vectors(model):
    ids = [TOKEN_TO_ID[f"P{m}"] for m in range(LOW, HIGH + 1)]
    return model.transformer.wte.weight[ids].detach().numpy()


trained = GPT2LMHeadModel.from_pretrained("bachLLM")
torch.manual_seed(0)
untrained = GPT2LMHeadModel(trained.config)

learned = by_interval(pitch_vectors(trained))
random_ = by_interval(pitch_vectors(untrained))

print(f"Pitch similarity by distance, P{LOW}..P{HIGH}\n")
print(f"{'half-steps':>10}  {'interval':<13}{'trained':>8}{'untrained':>10}")
for d in range(1, 25):
    print(f"{d:>10}  {NAMES.get(d, ''):<13}{learned[d]:>8.3f}{random_[d]:>10.3f}")


def bump(s, d):
    """How much more alike distance d is than its neighbours d-1 and d+1."""
    return s[d] - (s[d - 1] + s[d + 1]) / 2


print("\nPeaks (above the two neighbouring distances):")
for d in (7, 12, 19, 24):
    print(f"  {NAMES[d]:<13} trained {bump(learned, d):+.3f}   untrained {bump(random_, d):+.3f}")

# Nearest neighbours: for each pitch, which other pitches does the model
# treat as most alike?
v = pitch_vectors(trained)
v = v / np.linalg.norm(v, axis=1, keepdims=True)
sim = v @ v.T
octave_rank = []
for i in range(len(v)):
    order = [j for j in np.argsort(-sim[i]) if j != i]
    for j in (i - 12, i + 12):
        if 0 <= j < len(v):
            octave_rank.append(order.index(j) + 1)
print(f"\nOctave partner's rank among the {len(v) - 1} other pitches "
      f"(1 = most alike): median {int(np.median(octave_rank))}, "
      f"in top 3 for {100 * np.mean(np.array(octave_rank) <= 3):.0f}% of pitches")
