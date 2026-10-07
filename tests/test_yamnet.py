import numpy as np
import pytest
import torch

from vocal2midi import yamnet


def test_mel_matrix_matches_tensorflow_layout():
    m = yamnet.mel_matrix()
    assert m.shape == (257, 64)
    assert np.all(m[0] == 0)  # no DC bin
    assert np.all(m.max(0) > 0.5) and np.all(m >= 0)


@pytest.mark.parametrize("seconds, patches", [(0.1, 1), (0.975, 1), (1.455, 2), (24.0, 49)])
def test_patches(seconds, patches):
    x = yamnet.log_mel_patches(np.zeros(int(seconds * 16000), np.float32))
    assert x.shape == (patches, 96, 64)
    assert np.allclose(x, np.log(0.001))


def test_network_shapes():
    model = yamnet.YAMNet().eval()
    with torch.inference_mode():
        scores, emb = model(torch.zeros(3, 96, 64))
    assert scores.shape == (3, 521) and emb.shape == (3, 1024)


def test_same_padding_like_tensorflow():
    x = torch.zeros(1, 1, 96, 64)
    assert yamnet._same_pad(x, 2).shape[-2:] == (97, 65)  # odd padding goes after
    assert yamnet._same_pad(x, 1).shape[-2:] == (98, 66)


@pytest.mark.skipif(not yamnet.DEFAULT_WEIGHTS.exists(), reason="models/yamnet.h5 not downloaded")
def test_released_weights():
    model = yamnet.get_model()
    silence = yamnet.classify(np.zeros(16000, np.float32), model)[0]
    assert silence.argmax() == 494  # "Silence"
    rng = np.random.default_rng(0)
    noise = yamnet.classify(0.3 * rng.standard_normal(16000).astype(np.float32), model)[0]
    assert noise.argmax() in (507, 509, 514, 515)  # Noise, Static, White noise, Pink noise
