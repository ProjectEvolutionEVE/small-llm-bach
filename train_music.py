# train_music.py
#
# Trains the same GPT-2 architecture from random numbers on Bach chorale
# tokens (built by music_data.py). Only the vocabulary differs from train.py:
# ~75 music tokens instead of 50,257 word pieces.
#
# Run: .venv/bin/python train_music.py
# Output: bachLLM/ (the trained model)

import json
import random

from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

from music_tokens import VOCAB, PAD_ID, BOS_ID, EOS_ID

BLOCK = 512            # tokens per training chunk (about 8 bars of 4/4)
EPOCHS = 2             # ~7 min each on the Intel iMac's CPU

# -----------------------
# Data: pack all pieces end to end, then cut into equal chunks
# -----------------------
def packed(path, shuffle):
    seqs = json.load(open(path))
    if shuffle:
        random.Random(0).shuffle(seqs)
    stream = [t for s in seqs for t in s]
    chunks = [stream[i:i + BLOCK] for i in range(0, len(stream) - BLOCK + 1, BLOCK)]
    return Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": [[1] * BLOCK] * len(chunks),
        "labels": chunks,
    })

train_ds = packed("music_data/train.json", shuffle=True)
val_ds = packed("music_data/val.json", shuffle=False)
print(f"train chunks: {len(train_ds)}, val chunks: {len(val_ds)}")

# -----------------------
# Model (random start, no pretrained weights)
# -----------------------
config = GPT2Config(
    vocab_size=len(VOCAB),
    n_positions=BLOCK,
    n_ctx=BLOCK,
    n_embd=384,
    n_layer=6,
    n_head=6,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    pad_token_id=PAD_ID,
)
model = GPT2LMHeadModel(config)
print(f"parameters: {model.num_parameters():,}")

# -----------------------
# Training
# -----------------------
args = TrainingArguments(
    output_dir="bachLLM",
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

model.save_pretrained("bachLLM")
print("final val:", trainer.evaluate())
