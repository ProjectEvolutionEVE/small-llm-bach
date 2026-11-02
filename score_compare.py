# score_compare.py
#
# Draws the model's score and Bach's real score side by side in one HTML page,
# using Verovio (runs offline, no internet).
#
# Run: .venv/bin/python score_compare.py [name]     (default: bwv4)
# Reads output/<name>_model.musicxml and output/<name>_bach.musicxml
# Writes output/<name>_compare.html

import sys
from pathlib import Path

import verovio

NAME = sys.argv[1] if len(sys.argv) > 1 else "bwv4"
OUT_DIR = Path("output")


def render(path):
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 1400,
        "pageHeight": 60000,
        "adjustPageHeight": True,
        "scale": 40,
        "breaks": "auto",
        "header": "none",
        "footer": "none",
    })
    tk.loadFile(str(path))
    return "".join(tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1))


model_svg = render(OUT_DIR / f"{NAME}_model.musicxml")
bach_svg = render(OUT_DIR / f"{NAME}_bach.musicxml")

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{NAME}: Small LLM vs Bach</title>
<style>
  body {{ margin: 0; padding: 16px; background: #fff; color: #1f2328;
         font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.note {{ margin: 0 0 16px; color: #57606a; max-width: 900px; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .col {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; min-width: 0; }}
  .col h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .model h2 {{ color: #1f6feb; }}
  .bach h2 {{ color: #1a7f37; }}
  .col svg {{ width: 100%; height: auto; display: block; }}
  @media (max-width: 800px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>{NAME}: Small LLM vs Bach</h1>
<p class="note">Both start with the same opening bars by Bach (black notes). Coloured notes are
the continuation: blue = predicted by the Small LLM, green = what Bach wrote. Bar lines are drawn
exactly where each one placed them, so odd time signatures in the blue score are bars the model
miscounted.</p>
<div class="cols">
  <div class="col model"><h2>Small LLM's prediction</h2>{model_svg}</div>
  <div class="col bach"><h2>Bach's real continuation</h2>{bach_svg}</div>
</div>
</body>
</html>
"""

out = OUT_DIR / f"{NAME}_compare.html"
out.write_text(html)
print("wrote", out)
