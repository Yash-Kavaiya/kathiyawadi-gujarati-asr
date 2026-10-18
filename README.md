# kathiyawadi-gujarati-asr

Open-source Gujarati speech recognition with a focus on the **Kathiyawadi (Saurashtra) dialect**:
data pipeline, pseudo-labelling, fine-tuning (IndicConformer + Whisper), and a reproducible
benchmark against every open Gujarati ASR model.

## Layout
| Path | What |
|---|---|
| `kgasr/` | Library: text normaliser, WER/CER + bootstrap CIs, manifests/splits, agreement filter, model wrappers |
| `data_pipeline/` | download → segment → LID → pseudo-label → filter → splits |
| `train/nemo/` | Fine-tune `ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large` (primary) |
| `train/whisper/` | Fine-tune `openai/whisper-large-v3-turbo` (comparison) |
| `lm/` | KenLM n-gram for CTC decoding |
| `eval/` | Export public test sets, run all baselines, produce the paper table |
| `paper/` | Paper outline / LaTeX |

All transcripts are scored through **one** normaliser (`kgasr/normalize.py`), so the numbers can be compared across models.

## Setup
```bash
pip install -e ".[dev]"                 # library + tests (any machine)
pytest
# Cloud GPU box (A100/H100):
pip install -e ".[pipeline,train,nemo]" speechbrain
huggingface-cli login                   # request access to the gated AI4Bharat repos first
```

## Runbook
```bash
# 1. Data (sources.txt: your channels / permissioned channels only)
cp configs/sources.example.txt configs/sources.txt
python data_pipeline/download.py --sources configs/sources.txt --out data/raw
#    Own recordings: put them in data/raw/audio/<speaker_id>/*.wav (16 kHz mono)
python data_pipeline/segment.py --raw data/raw --out data/segments
python data_pipeline/lid_filter.py --manifest data/segments/manifest.jsonl --out data/segments/manifest.gu.jsonl

# 2. Gold Kathiyawadi set: human transcripts → data/manifests/gold.jsonl
#    (audio_filepath, duration, text, speaker, channel), then split by speaker/channel:
python data_pipeline/make_splits.py --inputs data/manifests/gold.jsonl --out data/manifests

# 3. Pseudo-labels (two models), choose threshold on dev, filter
python data_pipeline/pseudolabel.py --manifest data/segments/manifest.gu.jsonl \
    --kind nemo --model ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large --name conformer
python data_pipeline/pseudolabel.py --manifest data/segments/manifest.gu.jsonl \
    --kind whisper --model vasista22/whisper-gujarati-medium --name whisper   # or IndicWhisper path
#    (run both pseudolabel commands on kathiyawadi_dev.jsonl too, then:)
python data_pipeline/filter.py --manifest data/manifests/kathiyawadi_dev.jsonl \
    --primary hyp_conformer --secondary hyp_whisper --sweep
python data_pipeline/filter.py --manifest data/segments/manifest.gu.jsonl \
    --primary hyp_conformer --secondary hyp_whisper --max-cer 0.12 --out data/manifests/pseudo.jsonl
python data_pipeline/make_splits.py --inputs data/manifests/gold.jsonl --out data/manifests \
    --train-extra data/manifests/pseudo.jsonl                                  # leak check

# 4. Train
python train/nemo/train.py --config train/nemo/finetune_indicconformer.yaml
accelerate launch train/whisper/finetune.py --config train/whisper/config.yaml

# 5. Noisy-student: re-run step 3 with the new model as --kind nemo --model runs/conformer-gu-r1/best.nemo,
#    retrain (r2, r3) until dev WER stops improving.

# 6. Evaluate everything
for s in fleurs kathbath kathbath_noisy indicvoices commonvoice; do python eval/export_public.py --set $s; done
python eval/run_benchmarks.py --config eval/baselines.yaml   # → eval/results/table.md, scores.json
```

## Data rights
- Use only YouTube channels you own or have **written permission** for.
- Get consent from every speaker in your own recordings.
- The released test set contains only audio you hold rights to. For any other YouTube data, publish only video IDs and timestamps.

## License
The code is Apache-2.0. Model weights and the dataset are licensed separately on release; each model/dataset card states its license.
