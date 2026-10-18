"""Fine-tune a NeMo hybrid CTC/RNNT model (IndicConformer-gu) on our manifests.

  python train/nemo/train.py --config train/nemo/finetune_indicconformer.yaml

Manifest `text` is normalised on the fly (spelling kept, punctuation stripped)
into a temp copy, because the AI4Bharat tokenizer has no punctuation tokens.
"""

import argparse
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from kgasr.asr import restore_nemo
from kgasr.manifest import read_manifest, write_manifest
from kgasr.normalize import normalize


def prepared(paths, tmpdir):
    """Write normalised copies; repeated paths stay repeated (that is the upsampling)."""
    out, cache = [], {}
    for p in paths:
        if p not in cache:
            rows = [dict(r, text=normalize(r["text"], remove_nukta=False, candrabindu_to_anusvara=False))
                    for r in read_manifest(p) if r.get("text")]
            dst = Path(tmpdir) / f"{len(cache)}_{Path(p).name}"
            write_manifest(dst, rows)
            cache[p] = str(dst)
        out.append(cache[p])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="kgasr_nemo_")

    model = restore_nemo(cfg["init_from"])
    train_paths = prepared(cfg["train_manifests"], tmp)
    dev_path = prepared([cfg["dev_manifest"]], tmp)[0]

    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = train_paths
        model.cfg.train_ds.batch_size = cfg["batch_size"]
        model.cfg.train_ds.max_duration = cfg["max_duration"]
        model.cfg.train_ds.min_duration = cfg["min_duration"]
        model.cfg.train_ds.num_workers = cfg["num_workers"]
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.is_tarred = False
        model.cfg.train_ds.tarred_audio_filepaths = None
        if cfg.get("speed_perturb"):
            model.cfg.train_ds.augmentor = {"speed": {"prob": 0.5, "sr": 16000, "resample_type": "kaiser_fast",
                                                      "min_speed_rate": 0.9, "max_speed_rate": 1.1}}
        model.cfg.validation_ds.manifest_filepath = dev_path
        model.cfg.validation_ds.batch_size = cfg["batch_size"]
        model.cfg.validation_ds.num_workers = cfg["num_workers"]
        model.cfg.optim = OmegaConf.create(cfg["optim"])

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_optimization(model.cfg.optim)

    ckpt = ModelCheckpoint(dirpath=cfg["output_dir"], monitor="val_wer", mode="min", save_top_k=5,
                           filename="{step}-{val_wer:.4f}")
    trainer = pl.Trainer(default_root_dir=cfg["output_dir"], callbacks=[ckpt], **cfg["trainer"])
    model.set_trainer(trainer)
    trainer.fit(model)

    best = type(model).load_from_checkpoint(ckpt.best_model_path)
    best.save_to(f"{cfg['output_dir']}/best.nemo")
    print("saved", f"{cfg['output_dir']}/best.nemo", "val_wer", ckpt.best_model_score)


if __name__ == "__main__":
    main()
