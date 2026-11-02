# music_tokens.py
#
# The music "tokenizer": turns four-voice scores into token IDs and back.
# Built here from scratch, no downloaded vocabulary.
#
# Time is cut into sixteenth-note steps. Each step writes 4 tokens, one per
# voice, always in the order Soprano, Alto, Tenor, Bass:
#   P60  = a new note starts (MIDI pitch 60 = middle C)
#   HOLD = the voice keeps holding its previous note
#   REST = the voice is silent
# BAR marks each bar line. BOS / EOS mark the start and end of a piece.

STEPS_PER_QUARTER = 4          # sixteenth-note grid
VOICES = 4                     # S, A, T, B
LOW, HIGH = 28, 96             # MIDI pitch range covered by the vocabulary

SPECIAL = ["PAD", "BOS", "EOS", "BAR", "HOLD", "REST"]
PITCHES = [f"P{m}" for m in range(LOW, HIGH + 1)]
VOCAB = SPECIAL + PITCHES

TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID["PAD"]
BOS_ID = TOKEN_TO_ID["BOS"]
EOS_ID = TOKEN_TO_ID["EOS"]
BAR_ID = TOKEN_TO_ID["BAR"]


# -----------------------
# Score -> tokens
# -----------------------
def score_to_grid(score):
    """Return a list of bars; each bar is a list of steps; each step is
    4 values (MIDI pitch, "HOLD" or None for rest). Returns None if the
    score is not a clean four-voice piece."""
    from music21 import chord

    parts = score.parts
    if len(parts) != VOICES:
        return None

    # Note spans per voice: (start_step, end_step, midi)
    spans = []
    for part in parts:
        voice = []
        for el in part.stripTies().flatten().notesAndRests:
            start = round(float(el.offset) * STEPS_PER_QUARTER)
            end = round(float(el.offset + el.quarterLength) * STEPS_PER_QUARTER)
            if end <= start:
                continue  # grace notes
            if el.isRest:
                continue
            if isinstance(el, chord.Chord):
                midi = max(p.midi for p in el.pitches)
            else:
                midi = el.pitch.midi
            voice.append((start, end, midi))
        spans.append(voice)

    # Bar lines from the soprano part
    measures = list(parts[0].getElementsByClass("Measure"))
    if not measures:
        return None

    bars = []
    for m in measures:
        bar_start = round(float(m.offset) * STEPS_PER_QUARTER)
        bar_len = round(float(m.duration.quarterLength) * STEPS_PER_QUARTER)
        if bar_len <= 0:
            continue
        bar = []
        for t in range(bar_start, bar_start + bar_len):
            step = []
            for voice in spans:
                value = None
                for start, end, midi in voice:
                    if start <= t < end:
                        value = midi if start == t else "HOLD"
                        break
                step.append(value)
            bar.append(step)
        bars.append(bar)
    return bars


def grid_to_ids(bars, shift=0):
    """Encode a grid as token IDs, transposed by `shift` semitones.
    Returns None if the transposition leaves the vocabulary's range."""
    ids = [BOS_ID]
    for bar in bars:
        ids.append(BAR_ID)
        for step in bar:
            for value in step:
                if value is None:
                    ids.append(TOKEN_TO_ID["REST"])
                elif value == "HOLD":
                    ids.append(TOKEN_TO_ID["HOLD"])
                else:
                    midi = value + shift
                    if not LOW <= midi <= HIGH:
                        return None
                    ids.append(TOKEN_TO_ID[f"P{midi}"])
    ids.append(EOS_ID)
    return ids


# -----------------------
# Tokens -> notes
# -----------------------
def ids_to_notes(ids, with_bars=False):
    """Decode token IDs into notes per voice: lists of (start_step, length, midi).
    Tolerant of a model that drifts: tokens are dealt to voices in turn,
    and each BAR starts the turn again at the soprano.
    With with_bars=True, also returns the step where each bar starts and
    the total number of steps."""
    notes = [[] for _ in range(VOICES)]
    current = [None] * VOICES   # [start, length, midi] of the note each voice holds
    voice = 0
    step = 0
    bars = []

    def close(v):
        if current[v] is not None:
            notes[v].append(tuple(current[v]))
            current[v] = None

    for i in ids:
        tok = ID_TO_TOKEN.get(int(i), "PAD")
        if tok in ("PAD", "BOS"):
            continue
        if tok == "EOS":
            break
        if tok == "BAR":
            if voice != 0:      # unfinished step: count it as complete
                step += 1
                voice = 0
            bars.append(step)
            continue

        if tok == "HOLD":
            if current[voice] is not None:
                current[voice][1] += 1
        elif tok == "REST":
            close(voice)
        else:
            close(voice)
            current[voice] = [step, 1, int(tok[1:])]

        voice += 1
        if voice == VOICES:
            voice = 0
            step += 1

    for v in range(VOICES):
        close(v)
    if voice != 0:
        step += 1
    if with_bars:
        return notes, bars, step
    return notes
