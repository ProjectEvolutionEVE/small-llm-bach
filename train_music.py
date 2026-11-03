# train_music.py
#
# Trains the same GPT-2 architecture from random numbers on Bach chorale
# tokens (built by music_data.py). Only the vocabulary differs from train.py:
# ~75 music tokens instead of 50,257 word pieces.
#
# Run: .venv/bin/python train_music.py [OUT_DIR] [--position] [--physics] [--half]
#   --position  add voice and place-in-bar to each token (experiment 2)
#   --physics   add the physics of each pitch to each token (experiment 2)
#   --half      train on half the training chorales
# Output: OUT_DIR (default bachLLM/, the trained model)

import json
import random
import sys

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from music_model import BachGPT, make_config

BLOCK = 512            # tokens per training chunk (about 8 bars of 4/4)
EPOCHS = 2             # ~7 min each on the Intel iMac's CPU

flags = [a for a in sys.argv[1:] if a.startswith("--")]
names = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT_DIR = names[0] if names else "bachLLM"
HALF = "--half" in flags

# -----------------------
# Data: pack all pieces end to end, then cut into equal chunks
# -----------------------
def packed(path, shuffle, half=False):
    seqs = json.load(open(path))
    if half:   # keep the first half of the pieces (already in random order), all their keys
        groups = json.load(open("music_data/train_groups.json"))
        keep = (max(groups) + 1) // 2
        seqs = [s for s, g in zip(seqs, groups) if g < keep]
    if shuffle:
        random.Random(0).shuffle(seqs)
    stream = [t for s in seqs for t in s]
    chunks = [stream[i:i + BLOCK] for i in range(0, len(stream) - BLOCK + 1, BLOCK)]
    return Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": [[1] * BLOCK] * len(chunks),
        "labels": chunks,
    })

train_ds = packed("music_data/train.json", shuffle=True, half=HALF)
val_ds = packed("music_data/val.json", shuffle=False)
print(f"train chunks: {len(train_ds)}, val chunks: {len(val_ds)}")

# -----------------------
# Model (random start, no pretrained weights)
# -----------------------
config = make_config(BLOCK, "--position" in flags, "--physics" in flags)
model = BachGPT(config)
print(f"parameters: {model.num_parameters():,}")

# -----------------------
# Training
# -----------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    learning_rate=5e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

model.save_pretrained(OUT_DIR)
print("final val:", trainer.evaluate())
