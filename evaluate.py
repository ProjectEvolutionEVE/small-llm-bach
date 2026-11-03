# evaluate.py
#
# Scores one trained Bach model (any model dir that music_model.load opens),
# using the metrics defined in PREDICTIONS.md ("How we score"):
#   1. held-out loss and next-token accuracy on music_data/val.json
#   2. generation: 4 bars of a held-out chorale in, 8 BAR tokens' worth out,
#      several seeds per piece, sampled the same way for every model
#   3. bar accuracy, voice order, clashes, in-key, notes/bar, chords/bar for each
#      continuation, and for Bach's real continuation of the same pieces
# Nothing is trained. Writes one JSON file with everything (incl. token ids).
#
# Run: .venv/bin/python evaluate.py MODEL_DIR [--pieces N] [--seeds K] [--out PATH]
#   --pieces N  first N of the 20 held-out chorales (default 20)
#   --seeds K   seeds 0..K-1 per piece (default 3)
#   --out PATH  default results/<name of MODEL_DIR>.json

import argparse
import json
import time
from pathlib import Path

import torch
from music21 import corpus

from music_model import load
from music_tokens import (
    BAR_ID, EOS_ID, score_to_grid, grid_to_ids, ids_to_notes,
)

BLOCK = 512            # chunk length used in training
CONTEXT = 384          # tokens of history the model sees while generating
TEMPERATURE = 0.9
PROMPT_BARS = 4
NEW_BARS = 8           # the 1st new BAR opens bar 5; the 9th closes bar 12 (not kept)
MAX_NEW = 900
METRICS = ["bar_accuracy", "bar_whole_steps", "voice_order_errors", "clashes", "in_key",
           "notes_per_bar", "chords_per_bar", "bars_generated"]


# -----------------------
# 1. Held-out loss and accuracy
# -----------------------
def held_out(model):
    """Pack the held-out pieces end to end, cut into 512-token chunks
    (remainder dropped, like train_music.py), average loss over chunks."""
    seqs = json.load(open("music_data/val.json"))
    stream = [t for s in seqs for t in s]
    chunks = [stream[i:i + BLOCK] for i in range(0, len(stream) - BLOCK + 1, BLOCK)]
    ids = torch.tensor(chunks)
    loss_sum, right, total = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(ids), 8):
            batch = ids[i:i + 8]
            out = model(input_ids=batch, labels=batch)
            loss_sum += out.loss.item() * len(batch)     # chunks are equal size
            pred = out.logits[:, :-1].argmax(-1)
            right += (pred == batch[:, 1:]).sum().item()
            total += pred.numel()
    return loss_sum / len(ids), right / total, len(ids)


# -----------------------
# 2. Generation
# -----------------------
def generate(model, prompt, seeds):
    """Continue `prompt` once per seed, as one batch. No KV cache: each step
    feeds the last 384 tokens. Each sample has its own generator seeded with
    its seed. A sample that is done (EOS or its 8th new BAR) stops growing.
    Returns a list of (continuation ids, ended_early_by_EOS)."""
    n = len(seeds)
    seqs = torch.tensor([prompt] * n)
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    cont = [[] for _ in range(n)]
    bars = [0] * n
    done = [False] * n
    early = [False] * n
    with torch.no_grad():
        for _ in range(MAX_NEW):
            logits = model(input_ids=seqs[:, -CONTEXT:]).logits[:, -1]
            probs = torch.softmax(logits / TEMPERATURE, dim=-1)
            nxt = torch.stack([torch.multinomial(probs[i], 1, generator=gens[i])[0]
                               for i in range(n)])
            seqs = torch.cat([seqs, nxt[:, None]], dim=1)     # finished ones keep growing; ignored
            for i in range(n):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t == EOS_ID:
                    done[i], early[i] = True, True
                    continue
                if t == BAR_ID:
                    bars[i] += 1
                    if bars[i] == NEW_BARS + 1:
                        done[i] = True
                        continue
                cont[i].append(t)
            if all(done):
                break
    return list(zip(cont, early))


# -----------------------
# 3. Metrics
# -----------------------
def bar_sizes(cont, closed):
    """Number of step tokens in each bar of the continuation (text after
    each BAR). The last bar counts only if `closed`: it was ended by the
    8th BAR or by EOS, not cut off by the token limit."""
    sizes, cur, open_bar = [], 0, False
    for t in cont:
        if t == BAR_ID:
            if open_bar:
                sizes.append(cur)
            cur, open_bar = 0, True
        elif t > BAR_ID:        # a note/HOLD/REST token (ids 0-3 are markers)
            cur += 1
    if open_bar and closed:
        sizes.append(cur)
    return sizes


def metrics(prompt, cont, bar_len, scale_pcs, prompt_steps, closed):
    """All metrics for one continuation. `closed` says the last bar was
    ended (by the 8th BAR, which is not kept, or by EOS)."""
    # bar accuracy from the raw tokens
    segs = bar_sizes(cont, closed)
    # right length = a bar length Bach himself uses in this piece (full bars,
    # pickups, and the shorter bars he writes at phrase ends)
    ok = [n % 4 == 0 and n // 4 in bar_len for n in segs]
    n_bars = len(segs)

    # decode and keep only the steps after the prompt
    notes, bars, total = ids_to_notes(prompt + cont, with_bars=True)
    sounding = {}                       # step -> {voice: midi}
    for v, voice in enumerate(notes):
        for s, l, m in voice:
            for t in range(max(s, prompt_steps), s + l):
                sounding.setdefault(t, {})[v] = m
    order_den = order_bad = clash_den = clash_bad = 0
    for t in range(prompt_steps, total):
        sv = sounding.get(t, {})
        if len(sv) >= 2:
            clash_den += 1
            ms = list(sv.values())
            if any((a - b) % 12 in (1, 11) for i, a in enumerate(ms) for b in ms[i + 1:]):
                clash_bad += 1
        pairs = [(sv[v], sv[v + 1]) for v in range(3) if v in sv and v + 1 in sv]
        if pairs:
            order_den += 1
            if any(upper < lower for upper, lower in pairs):    # upper voice below lower one
                order_bad += 1

    onsets = [m for voice in notes for s, l, m in voice if s >= prompt_steps]
    in_key = sum(m % 12 in scale_pcs for m in onsets)

    # distinct chords per bar (bars as the decoder sees them)
    starts = [b for b in bars if b >= prompt_steps] + [total]
    chord_counts = []
    for a, z in zip(starts[:-1], starts[1:]):
        if z <= a:
            continue
        chords = {frozenset(m % 12 for m in sounding.get(t, {}).values())
                  for t in range(a, z)}
        chords.discard(frozenset())
        chord_counts.append(len(chords))

    return {
        "bar_whole_steps": sum(n % 4 == 0 for n in segs) / n_bars if n_bars else None,
        "bar_accuracy": sum(ok) / n_bars if n_bars else None,
        "voice_order_errors": order_bad / order_den if order_den else None,
        "clashes": clash_bad / clash_den if clash_den else None,
        "in_key": in_key / len(onsets) if onsets else None,
        "notes_per_bar": len(onsets) / n_bars if n_bars else None,
        "chords_per_bar": sum(chord_counts) / len(chord_counts) if chord_counts else None,
        "bars_generated": n_bars,
    }


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--pieces", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    out_path = Path(args.out or f"results/{Path(args.model_dir.rstrip('/')).name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model = load(args.model_dir)
    loss, acc, n_chunks = held_out(model)
    print(f"held-out loss {loss:.4f}  accuracy {acc:.4f}  ({n_chunks} chunks)", flush=True)

    seeds = list(range(args.seeds))
    paths = json.load(open("music_data/val_pieces.json"))[:args.pieces]
    pieces, model_rows, bach_rows = [], [], []
    gen_time = 0.0
    for p, path in enumerate(paths):
        original = corpus.parse(path)
        grid = score_to_grid(original)
        bar_len = sorted({len(b) for b in grid})          # every bar length Bach uses here
        key_ = original.analyze("key")
        scale_pcs = {pt.pitchClass for pt in key_.getScale().getPitches()}
        prompt_steps = sum(len(b) for b in grid[:PROMPT_BARS])

        # same prompt construction as generate_music.py
        prompt = grid_to_ids(grid[:PROMPT_BARS])[:-1]
        real = grid_to_ids(grid[:PROMPT_BARS + NEW_BARS])
        real_cont = real[len(prompt):-1]                     # Bach's bars 5-12, EOS dropped
        bach = metrics(prompt, real_cont, bar_len, scale_pcs, prompt_steps, closed=True)
        bach_rows.append(bach)

        t0 = time.time()
        results = generate(model, prompt, seeds)
        gen_time += time.time() - t0
        conts = []
        for seed, (cont, early) in zip(seeds, results):
            m = metrics(prompt, cont, bar_len, scale_pcs, prompt_steps, closed=len(cont) < MAX_NEW)
            m["eos_early"] = early
            model_rows.append(m)
            conts.append({"seed": seed, "metrics": m, "ids": cont})
        pieces.append({"piece": path, "key": str(key_), "bar_len": bar_len,
                       "prompt_steps": prompt_steps, "bach": {"metrics": bach, "ids": real_cont},
                       "continuations": conts})
        print(f"[{p + 1}/{len(paths)}] {path}: " + ", ".join(
            f"bar_acc {c['metrics']['bar_accuracy']}" for c in conts), flush=True)

    keys = METRICS
    model_means = {k: mean(model_rows, k) for k in keys}
    model_means["eos_early"] = sum(r["eos_early"] for r in model_rows) / len(model_rows)
    bach_means = {k: mean(bach_rows, k) for k in keys}
    elapsed = time.time() - start
    result = {
        "model_dir": args.model_dir, "loss": loss, "accuracy": acc,
        "n_continuations": len(model_rows),
        "model_means": model_means, "bach_means": bach_means,
        "seconds": elapsed, "generation_seconds": gen_time,
        "settings": {"prompt_bars": PROMPT_BARS, "new_bars": NEW_BARS, "temperature": TEMPERATURE,
                     "context": CONTEXT, "max_new": MAX_NEW, "seeds": seeds},
        "pieces": pieces,
    }
    out_path.write_text(json.dumps(result))

    print(f"\n{args.model_dir}: loss {loss:.4f}, accuracy {acc:.4f}, "
          f"{len(model_rows)} continuations, {elapsed:.0f} s")
    print(f"{'metric':<20}{'model':>10}{'Bach':>10}")
    for k in keys + ["eos_early"]:
        a, b = model_means[k], bach_means.get(k)
        print(f"{k:<20}{a:>10.3f}{(f'{b:.3f}' if b is not None else '-'):>10}")
    print(f"wrote {out_path}")


main()
