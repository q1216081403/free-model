import numpy as np

from free_model import FreeModel, FreeModelConfig


def test_forward_shape():
    cfg = FreeModelConfig(input_dim=32, hidden_dims=[16], num_classes=4)
    model = FreeModel(cfg)
    x = np.random.randn(5, 32).astype(np.float32)
    out = model.forward(x)
    assert out.shape == (5, 4)


def test_save_load_roundtrip(tmp_path):
    cfg = FreeModelConfig()
    model = FreeModel(cfg)
    model.save(tmp_path / "ckpt")
    reloaded = FreeModel.load(tmp_path / "ckpt")

    x = np.random.randn(2, cfg.input_dim).astype(np.float32)
    assert np.allclose(model.forward(x), reloaded.forward(x), atol=1e-6)


def test_predict():
    cfg = FreeModelConfig(num_classes=3)
    model = FreeModel(cfg)
    x = np.random.randn(4, cfg.input_dim).astype(np.float32)
    preds = model.predict(x)
    assert preds.shape == (4,)
    assert preds.min() >= 0 and preds.max() < 3


def test_predict_proba_sums_to_one():
    cfg = FreeModelConfig(num_classes=5)
    model = FreeModel(cfg)
    x = np.random.randn(3, cfg.input_dim).astype(np.float32)
    proba = model.predict_proba(x)
    assert np.allclose(proba.sum(axis=-1), 1.0, atol=1e-6)
