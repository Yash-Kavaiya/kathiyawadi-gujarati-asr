# Paper outline

**Working title:** Saurashtra-ASR: Pseudo-Label Bootstrapping for Dialectal Gujarati Speech Recognition and the First Kathiyawadi Benchmark

**Target venues:** Interspeech 2027 (deadline usually early March), ICASSP 2028, or SLT / LREC-COLING. Workshops: VarDial, SIGUL, LoResLM. Post a preprint on arXiv when you submit.

## Contributions
1. **The Kathiyawadi test set.** It is about 6–10 h from 40+ speakers, transcribed by humans in two passes, with no speakers or channels in common with the training data. Report the inter-annotator CER.
2. **The pseudo-labelling recipe.** It combines agreement between two models (or with subtitles), a character-rate filter, and noisy-student rounds. Report the measured label error rate from the human-checked sample.
3. **The open model.** Report SOTA results on Kathiyawadi and parity or better on FLEURS, Kathbath and IndicVoices Gujarati, all with 95% CIs and paired-bootstrap p-values.

## Sections
1. Introduction: the dialect gap in Indic ASR
2. Related work: Vistaar/IndicWhisper, IndicConformer, MMS, Omnilingual ASR, noisy-student training, dialect ASR
3. Benchmark: how it was collected, the transcription guideline, inter-annotator agreement, statistics (hours, speakers, districts, gender)
4. Method: the pipeline diagram, the agreement filter, the noisy-student loop, the KenLM language model
5. Setup: data mix, baselines (`eval/baselines.yaml`), and one shared normaliser for every model
6. Results: the main table (`eval/results/table.md`)
7. Ablations:
   - amount of pseudo-labelled data (50/100/200/all h)
   - agreement threshold (output of `filter.py --sweep`)
   - each noisy-student round
   - with vs. without public data
   - with vs. without the LM
   - Conformer vs. Whisper
8. Error analysis: dialect words, code-mixing, named entities, Whisper hallucinations
9. Limitations and Ethics: consent, YouTube data rights, speaker privacy, coverage of districts
10. Reproducibility: code, configs, seeds, GPU-hours used

## Rules
- Every number in the paper must come from `eval/results/scores.json`. Never copy numbers by hand.
- Never tune anything on the test set. Choose thresholds and checkpoints on dev only.
