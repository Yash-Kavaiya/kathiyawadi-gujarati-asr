"""Uniform `transcribe(paths) -> list[str]` wrappers around the ASR models we use.

Used both for pseudo-labelling and for benchmarking, so every model is run the
same way. Heavy imports are inside the loaders so the module imports anywhere.
"""

from typing import Callable

Transcriber = Callable[[list[str]], list[str]]


def load_whisper(model_id: str, device: str = "cuda", batch_size: int = 16) -> Transcriber:
    """Any HF Whisper checkpoint: openai/whisper-large-v3, IndicWhisper, our fine-tunes."""
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device=device,
    )

    def run(paths):
        outs = pipe(paths, batch_size=batch_size,
                    generate_kwargs={"language": "gujarati", "task": "transcribe"})
        return [o["text"].strip() for o in outs]

    return run


def restore_nemo(model_id: str, map_location: str = "cpu"):
    """Load a NeMo ASR model from a local .nemo path or an HF repo id containing one."""
    from nemo.collections.asr.models import ASRModel

    if model_id.endswith(".nemo"):
        return ASRModel.restore_from(model_id, map_location=map_location)
    from huggingface_hub import hf_hub_download, list_repo_files
    fname = next(f for f in list_repo_files(model_id) if f.endswith(".nemo"))
    return ASRModel.restore_from(hf_hub_download(model_id, fname), map_location=map_location)


def load_nemo(model_id: str, device: str = "cuda", batch_size: int = 16, decoder: str = "rnnt") -> Transcriber:
    """NeMo hybrid CTC/RNNT models: AI4Bharat IndicConformer-gu or our fine-tuned .nemo file."""
    import torch

    model = restore_nemo(model_id, map_location=device)
    model.eval()
    if hasattr(model, "change_decoding_strategy") and hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoder_type=decoder)

    def run(paths):
        with torch.inference_mode():
            outs = model.transcribe(paths, batch_size=batch_size)
        if isinstance(outs, tuple):  # older NeMo returns (best, all)
            outs = outs[0]
        return [o.text if hasattr(o, "text") else o for o in outs]

    return run


def load_indicconformer_multilingual(model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                                     device: str = "cuda", decoder: str = "rnnt") -> Transcriber:
    """The 600M multilingual model ships as trust_remote_code; call is model(wav, lang, decoder).

    Verify the call signature against the model card after access is granted.
    """
    import torch
    import torchaudio
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def run(paths):
        texts = []
        for p in paths:
            wav, sr = torchaudio.load(p)
            wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.inference_mode():
                texts.append(model(wav, "gu", decoder))
        return texts

    return run


def load_mms(model_id: str = "facebook/mms-1b-all", device: str = "cuda") -> Transcriber:
    import torch
    import torchaudio
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    proc = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    proc.tokenizer.set_target_lang("guj")
    model.load_adapter("guj")

    def run(paths):
        texts = []
        for p in paths:
            wav, sr = torchaudio.load(p)
            wav = torchaudio.functional.resample(wav.mean(0), sr, 16000)
            inp = proc(wav.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            with torch.inference_mode():
                ids = model(**inp).logits.argmax(-1)[0]
            texts.append(proc.decode(ids))
        return texts

    return run


LOADERS = {
    "whisper": load_whisper,
    "nemo": load_nemo,
    "indicconformer_multi": load_indicconformer_multilingual,
    "mms": load_mms,
}


def load(kind: str, model_id: str, **kw) -> Transcriber:
    return LOADERS[kind](model_id, **kw)
