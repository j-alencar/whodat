import numpy as np
import pytest

from whodat.embed import EMBEDDERS, embed_all
from whodat.media import SAMPLE_RATE


@pytest.fixture(scope="module", params=sorted(EMBEDDERS))
def embedder(request):
    return EMBEDDERS[request.param]()


def test_audio_shorter_than_half_a_second_embeds_as_nan(embedder):
    noise = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE // 4).astype("float32")

    assert np.isnan(embed_all(embedder, [noise])).all()


def test_same_audio_embeds_to_the_same_direction(embedder):
    noise = np.random.default_rng(0).normal(0, 0.1, 2 * SAMPLE_RATE).astype("float32")

    rows = embed_all(embedder, [noise, noise.copy()])

    units = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    assert units[0] @ units[1] == pytest.approx(1.0, abs=1e-4)
