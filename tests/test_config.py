import pytest

from respiratory_xray.config import ConfigError, config_from_dict, load_config


def test_default_config_loads(config_dir):
    cfg = load_config(config_dir)
    assert cfg.model.num_classes == len(cfg.data.classes)
    assert cfg.tracking.wandb_enabled is False
    assert cfg.data.allow_download is False


def test_roundtrip_reconstructs_preprocessing_and_class_order(config_dir):
    cfg = load_config(config_dir)
    rebuilt = config_from_dict(cfg.to_dict())
    assert rebuilt == cfg
    assert rebuilt.fingerprint() == cfg.fingerprint()
    sig = rebuilt.preprocessing_signature()
    assert sig["classes"] == list(cfg.data.classes)
    assert sig["preprocessing"]["image_size"] == 224


def test_class_order_changes_fingerprint(config_dir):
    cfg = load_config(config_dir)
    d = cfg.to_dict()
    d["data"]["classes"] = list(reversed(d["data"]["classes"]))
    assert config_from_dict(d).fingerprint() != cfg.fingerprint()


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"num_classes": 3}},
        {"data": {"classes": ["A", "A"]}},
        {"data": {"allow_download": True}},
        {"data": {"split_manifest": "C:/abs/path.csv"}},
        {"data": {"split_manifest": "https://example.org/x.csv"}},
        {"train": {"lr": 5}},
        {"train": {"unknown_key": 1}},
        {"seeds": {"global_seed": -1}},
        {"evaluation": {"selection_metric": "f1"}},
        {"data": {"preprocessing": {"image_size": 10}}},
        {"tracking": {"api_token": "x"}},
    ],
)
def test_invalid_values_rejected(config_dir, override):
    with pytest.raises(ConfigError):
        load_config(config_dir, overrides=override)


def test_local_example_not_in_experiment(config_dir):
    cfg = load_config(config_dir).to_dict()
    text = str(cfg)
    assert "data_root" not in text and "wandb_api_key" not in text
