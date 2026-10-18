"""Fine-tune Whisper (default: large-v3-turbo) on the mixed Gujarati manifests.

  accelerate launch train/whisper/finetune.py --config train/whisper/config.yaml

Audio is read and featurised in the collator, so there is no multi-hundred-GB
feature cache on disk. Manifest weights are integer upsampling factors.
"""

import argparse
import random

import soundfile as sf
import torch
import yaml
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

from kgasr.manifest import read_manifest
from kgasr.metrics import cer, wer
from kgasr.normalize import normalize


def train_text(t: str) -> str:
    # Keep spelling (nukta, candrabindu) in training targets; only eval folds them.
    return normalize(t, remove_nukta=False, candrabindu_to_anusvara=False)


def load_mix(entries, seed):
    rows = []
    for e in entries:
        part = [r for r in read_manifest(e["path"]) if r.get("text") and r["duration"] <= 30]
        rows += part * int(e.get("weight", 1))
        print(f"{e['path']}: {len(part)} segs x{e.get('weight', 1)}")
    random.Random(seed).shuffle(rows)
    return rows


class Collator:
    def __init__(self, processor, decoder_start_id):
        self.p = processor
        self.start = decoder_start_id

    def __call__(self, rows):
        audio = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rows]
        feats = self.p.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        lab = self.p.tokenizer([train_text(r["text"]) for r in rows], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.start).all():  # the model re-adds the start token
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))

    processor = WhisperProcessor.from_pretrained(cfg["model"], language="gujarati", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=torch.float32)
    model.generation_config.language = "gujarati"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = cfg.get("mask_time_prob", 0.05)
    model.config.mask_feature_prob = cfg.get("mask_feature_prob", 0.05)

    train_rows = load_mix(cfg["train"], cfg.get("seed", 0))
    dev_rows = read_manifest(cfg["dev"])[: cfg.get("max_dev", 2000)]

    def compute_metrics(pred):
        ids = pred.predictions
        lab = pred.label_ids
        lab[lab == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        refs = processor.batch_decode(lab, skip_special_tokens=True)
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=225,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to=cfg.get("report_to", ["tensorboard"]),
        **cfg["training_args"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=dev_rows,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/best")
    processor.save_pretrained(cfg["output_dir"] + "/best")


if __name__ == "__main__":
    main()
