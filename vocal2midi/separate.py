import time
from functools import lru_cache

import torch
from demucs.apply import BagOfModels, apply_model
from demucs.pretrained import get_model

from .audio import load_audio, write_wav


@lru_cache(maxsize=2)
def _vocals_model(name: str):
    """Load a demucs model, keeping only the sub-model that produces vocals.

    htdemucs_ft is a bag of 4 models, one specialised per stem. demucs runs all
    four even when you only want vocals; running just the vocals expert gives
    bit-identical vocals in 1/4 of the time.
    """
    model = get_model(name)
    if isinstance(model, BagOfModels):
        vi = model.sources.index("vocals")
        weights = [w[vi] for w in model.weights]
        if weights.count(0) == len(weights) - 1:
            model = model.models[weights.index(max(weights))]
    model.eval()
    return model


def separate_vocals(song, vocals_path, instrumental_path, model_name="htdemucs_ft",
                    device="cpu", overlap=0.25, max_seconds=None, progress=None):
    """`max_seconds` separates just the start (for a quick preview); `progress(fraction)`
    is called as demucs works through the song."""
    t0 = time.time()
    model = _vocals_model(model_name)
    sr = model.samplerate
    mix = torch.from_numpy(load_audio(song, sr, channels=model.audio_channels, max_seconds=max_seconds))

    def report(info):
        if progress and info.get("state") == "end" and info.get("audio_length"):
            progress(min(1.0, (info["segment_offset"] + sr * 10) / info["audio_length"]))

    # Same normalisation demucs' own CLI applies.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        stems = apply_model(model, ((mix - mean) / std)[None], device=device, shifts=0, overlap=overlap,
                            split=True, progress=progress is None, callback=report if progress else None)[0]
    vocals = stems[model.sources.index("vocals")] * std + mean

    write_wav(vocals_path, vocals.numpy(), sr)
    write_wav(instrumental_path, (mix - vocals).numpy(), sr)
    print(f"  separated in {time.time() - t0:.1f}s on {device}")
