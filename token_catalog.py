# token_catalog.py
#
# Lists every token in the music vocabulary (music_tokens.py) with its ID,
# what it means, and how often it appears in the training data.
#
# Run: .venv/bin/python token_catalog.py
# Writes output/token_catalog.html

import json
from collections import Counter
from pathlib import Path

from music_tokens import VOCAB, TOKEN_TO_ID

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MEANING = {
    "PAD": "Filler, never trained on (unused: chunks are packed full)",
    "BOS": "Beginning of a piece",
    "EOS": "End of a piece",
    "BAR": "Bar line (takes no time)",
    "HOLD": "This voice keeps holding its note for one more sixteenth",
    "REST": "This voice is silent for one sixteenth",
}

seqs = json.load(open("music_data/train.json"))
counts = Counter(t for s in seqs for t in s)
total = sum(counts.values())

rows = []
for tok in VOCAB:
    i = TOKEN_TO_ID[tok]
    if tok in MEANING:
        meaning = MEANING[tok]
    else:
        midi = int(tok[1:])
        meaning = f"Start note {NAMES[midi % 12]}{midi // 12 - 1} (MIDI {midi})"
    n = counts.get(i, 0)
    rows.append((i, tok, meaning, n, 100 * n / total))

used = sum(1 for r in rows if r[3] > 0)
body = "\n".join(
    f"<tr class='{'zero' if n == 0 else ''}'><td>{i}</td><td><code>{tok}</code></td>"
    f"<td>{meaning}</td><td class='num'>{n:,}</td><td class='num'>{pct:.2f}%</td></tr>"
    for i, tok, meaning, n, pct in rows
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bach Token Catalog</title>
<style>
  body {{ margin: 0; padding: 16px; background: #fff; color: #1f2328;
         font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p {{ margin: 0 0 12px; color: #57606a; max-width: 760px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 760px; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eaeef2; }}
  th {{ background: #f6f8fa; position: sticky; top: 0; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.zero td {{ color: #8c959f; }}
  code {{ font: 13px ui-monospace, Menlo, monospace; }}
</style>
</head>
<body>
<h1>Bach model: token catalog</h1>
<p>{len(VOCAB)} tokens in all: {len(MEANING)} markers and
{len(VOCAB) - len(MEANING)} pitches. {used} appear in the training data
({total:,} tokens, all 12 keys); grey rows never do. Every sixteenth-note step is 4 tokens,
one each for soprano, alto, tenor and bass.</p>
<table>
<thead><tr><th>ID</th><th>Token</th><th>Meaning</th><th class="num">Count</th><th class="num">Share</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""

Path("output").mkdir(exist_ok=True)
Path("output/token_catalog.html").write_text(html)
print(f"{len(VOCAB)} tokens, {used} used; wrote output/token_catalog.html")
for i, tok, meaning, n, pct in sorted(rows, key=lambda r: -r[3])[:6]:
    print(f"  {tok:5} {pct:5.1f}%")
