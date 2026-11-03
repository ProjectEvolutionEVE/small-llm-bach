# music_model.py
#
# The Bach model with optional extra facts added to each token (experiment 2,
# see PREDICTIONS.md). Same GPT-2 architecture and the same tokens; the only
# change is what gets added to each token's vector before the model reads it:
#
#   use_position: which voice the token is for (S, A, T, B) and which
#                 sixteenth of the bar it falls on. Counting only, worked out
#                 from the token stream itself.
#   use_physics:  for pitch tokens, fixed facts set by physics (approved by
#                 Eric 2026-11-02): note name, octave, frequency, and where the
#                 first 8 overtones land. These numbers never change during
#                 training; the model only learns how to use them.
#
# With both off it is the plain GPT-2 model used in the first test.
# load(model_dir) opens either kind, including the original bachLLM/.

import math

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from music_tokens import VOCAB, BAR_ID, BOS_ID, EOS_ID, PAD_ID, VOICES

MAX_STEP = 32          # sixteenths per bar we count up to


def physics_table():
    """One row of fixed numbers per token. Non-pitch tokens get zeros."""
    rows = []
    for tok in VOCAB:
        row = [0.0] * 26
        if tok.startswith("P") and tok[1:].isdigit():
            midi = int(tok[1:])
            row[midi % 12] = 1.0                                   # note name
            freq = 440 * 2 ** ((midi - 69) / 12)
            row[12] = (math.log2(freq) - math.log2(440)) / 3       # frequency (log Hz)
            row[13] = (midi // 12 - 5) / 3                         # octave
            for k in range(1, 9):                                  # overtones k*f
                above = 12 * math.log2(k)                          # half-steps above
                row[14 + round(midi + above) % 12] += 1 / k        # louder ones count more
        rows.append(row)
    return torch.tensor(rows)


def positions(ids):
    """For each token: voice slot (1-4) and sixteenth of the bar (1-MAX_STEP).
    0 means unknown: markers, and anything before the first BAR in view."""
    voice = torch.zeros_like(ids)
    step = torch.zeros_like(ids)
    for b, row in enumerate(ids.tolist()):
        count = None
        for i, t in enumerate(row):
            if t == BAR_ID:
                count = 0
            elif t in (BOS_ID, EOS_ID, PAD_ID):
                count = None
            elif count is not None:
                voice[b, i] = count % VOICES + 1
                step[b, i] = min(count // VOICES, MAX_STEP - 1) + 1
                count += 1
    return voice, step


class BachGPT(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        d = config.n_embd
        if getattr(config, "use_position", False):
            self.voice_emb = nn.Embedding(VOICES + 1, d)
            self.step_emb = nn.Embedding(MAX_STEP + 1, d)
        if getattr(config, "use_physics", False):
            self.register_buffer("physics", physics_table(), persistent=False)
            self.physics_proj = nn.Linear(self.physics.shape[1], d, bias=False)
        self.post_init()

    def token_vectors(self, input_ids):
        x = self.transformer.wte(input_ids)
        if getattr(self.config, "use_position", False):
            voice, step = positions(input_ids)
            x = x + self.voice_emb(voice) + self.step_emb(step)
        if getattr(self.config, "use_physics", False):
            x = x + self.physics_proj(self.physics[input_ids])
        return x

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                inputs_embeds=None, **kwargs):
        # labels and attention_mask are named so the Trainer passes them in
        if input_ids is not None:
            inputs_embeds = self.token_vectors(input_ids)
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                               labels=labels, **kwargs)


def make_config(block, use_position=False, use_physics=False):
    return GPT2Config(
        vocab_size=len(VOCAB), n_positions=block, n_ctx=block,
        n_embd=384, n_layer=6, n_head=6,
        bos_token_id=BOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
        use_position=use_position, use_physics=use_physics,
    )


def load(model_dir):
    return BachGPT.from_pretrained(model_dir).eval()
